import glob
import json
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
dataset = '30G'
files = glob.glob('/data/liuao/' + dataset + '_data/part-*.parquet')
dfs = []
for file in tqdm(files, desc="读取Parquet文件"):
    dfs.append(pq.read_table(file).to_pandas())
df_raw = pd.concat(dfs, ignore_index=True)

from ast import literal_eval

def parse_purchase_history(ph):
    if isinstance(ph, str):
        try:
            ph = json.loads(ph)
        except json.JSONDecodeError:
            try:
                ph = literal_eval(ph)
            except:
                return None
    if not isinstance(ph, dict):
        return None
    return {
        'user_name': ph.get('user_name'),
        'payment_method': ph.get('payment_method'),
        'payment_status': ph.get('payment_status'),
        'purchase_date': ph.get('purchase_date'),
        'items': ph.get('items', [])
    }

parsed = []
for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="解析 purchase_history"):
    rec = parse_purchase_history(row['purchase_history'])
    if rec:
        rec['order_id'] = row['id']
        parsed.append(rec)
parsed_df = pd.DataFrame(parsed)

exploded = parsed_df.explode('items').reset_index(drop=True)
exploded['item_id'] = exploded['items'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)

with open('products.json', encoding='utf-8') as f:
    catalog = json.load(f)['products']
catalog_df = pd.DataFrame(catalog)[['id', 'price', 'category']]

final = exploded.merge(catalog_df, left_on='item_id', right_on='id', how='left')
final = final[['order_id', 'user_name', 'payment_method', 'payment_status', 'purchase_date', 'item_id', 'price', 'category']]
final['purchase_date'] = pd.to_datetime(final['purchase_date'])

# 2. 商品类别关联规则挖掘
order_groups = final.groupby('order_id')['category'].apply(lambda x: x.dropna().unique().tolist())
te = TransactionEncoder()
te_ary = te.fit(order_groups).transform(order_groups)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)
freq_itemsets = apriori(df_trans, min_support=0.02, use_colnames=True)
rules_cat = association_rules(freq_itemsets, metric="confidence", min_threshold=0.3)

electronics = ['智能手机','笔记本电脑','平板电脑','智能手表','耳机','音响','相机','摄像机','游戏机']
rules_elec = rules_cat[
    rules_cat['antecedents'].apply(lambda x: any(e in x for e in electronics)) |
    rules_cat['consequents'].apply(lambda x: any(e in x for e in electronics))
]

# 3. 支付方式与商品类别的关联分析
final_pm = final.copy()
final_pm['cat_with_pay'] = final_pm['payment_method'] + "_PM__" + final_pm['category']
pay_groups = final_pm.groupby('order_id')['cat_with_pay'].apply(lambda x: x.dropna().unique().tolist())

te2 = TransactionEncoder()
te2_ary = te2.fit(pay_groups).transform(pay_groups)
df_pay = pd.DataFrame(te2_ary, columns=te2.columns_)
freq_pay = apriori(df_pay, min_support=0.01, use_colnames=True)

# 如果没有频繁项集，则返回空表格避免报错
if freq_pay.empty:
    rules_pay = pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift','leverage','conviction'])
else:
    rules_pay = association_rules(freq_pay, metric="confidence", min_threshold=0.5)

# 高价值商品 (>5000) 支付方式偏好
high_value = final[final['price'] > 5000]
hv_pref = high_value.groupby('order_id')['payment_method'].first().value_counts(normalize=True)

# 4. 时间序列模式挖掘
final_time = final.copy()
final_time['quarter'] = final_time['purchase_date'].dt.to_period('Q')
final_time['month'] = final_time['purchase_date'].dt.month
final_time['weekday'] = final_time['purchase_date'].dt.day_name()

quarter_counts = final_time.groupby('quarter').size()
month_counts = final_time.groupby('month').size()
weekday_counts = final_time.groupby('weekday').size().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

cat_month = final_time.groupby(['month','category']).size().unstack(fill_value=0)

from collections import Counter
seq = final_time.sort_values(['user_name','purchase_date']).groupby('user_name')['category'].apply(list)
pairs = Counter()
for seq_list in seq:
    for i in range(len(seq_list)-1):
        pairs[(seq_list[i], seq_list[i+1])] += 1
seq_df = pd.DataFrame([
    {'antecedent': k[0], 'consequent': k[1], 'count': v}
    for k, v in pairs.items()]
)

# 5. 退款模式分析
refund_df = final[final['payment_status'].isin(['已退款','部分退款'])]
refund_groups = refund_df.groupby('order_id')['category'].apply(lambda x: x.dropna().unique().tolist())

te3 = TransactionEncoder()
te3_ary = te3.fit(refund_groups).transform(refund_groups)
df_ref = pd.DataFrame(te3_ary, columns=te3.columns_)
freq_ref = apriori(df_ref, min_support=0.005, use_colnames=True)

if freq_ref.empty:
    rules_ref = pd.DataFrame(columns=['antecedents','consequents','support','confidence','lift','leverage','conviction'])
else:
    rules_ref = association_rules(freq_ref, metric="confidence", min_threshold=0.3)

# 6. 导出结果
freq_itemsets.to_csv('result/freq_itemsets.csv', index=False)
rules_cat.to_csv('result/rules_categories.csv', index=False)
rules_elec.to_csv('result/rules_electronics_related.csv', index=False)
rules_pay.to_csv('result/rules_payment_category.csv', index=False)
hv_pref.to_csv('result/hv_payment_pref.csv')
seq_df.to_csv('result/sequence_patterns.csv', index=False)

rules_ref.to_csv('result/rules_refund_patterns.csv', index=False)

print("任务完成，结果已导出。")
