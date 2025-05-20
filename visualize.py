import glob
import json
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans'],  # 优先使用开源字体
    'axes.unicode_minus': False
})
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = '1G'
files = glob.glob('/data/liuao/' + dataset + '_data/part-00000.parquet')
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


# 生成共现矩阵
category_pairs = final.groupby('order_id')['category'].agg(list).reset_index()['category']
te = TransactionEncoder()
te_ary = te.fit(category_pairs).transform(category_pairs)
df = pd.DataFrame(te_ary, columns=te.columns_)

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('商品类别共现热力图')
plt.savefig('./fig/fig1.png', dpi=300, bbox_inches='tight')
