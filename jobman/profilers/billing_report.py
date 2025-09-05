# pip install google-cloud-bigquery pandas
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from google.cloud import bigquery
import pandas as pd

# -------------------------
# 1) 可编辑的 Category 规则
# -------------------------
@dataclass
class CategoryRule:
    name: str
    service_contains: List[str] = None   # 在 service.description 中匹配的关键词（任一命中）
    sku_contains: List[str] = None       # 在 sku.description 中匹配的关键词（任一命中）

# 你可以按需增删规则；匹配按顺序第一个命中的规则为准
DEFAULT_CATEGORY_RULES: List[CategoryRule] = [
    # Storage
    CategoryRule(name="storage", service_contains=["Cloud Storage"]),
    CategoryRule(name="storage", sku_contains=["Persistent Disk", "PD Capacity", "Filestore", "Snapshot"]),
    # Network / Egress
    CategoryRule(name="network", service_contains=["Cloud CDN", "Traffic Director", "Cloud NAT"]),
    CategoryRule(name="network", sku_contains=["Egress", "Ingress", "Network Interconnect", "External IP"]),
    # Compute / Engine / GKE
    CategoryRule(name="compute", service_contains=["Compute Engine", "Kubernetes Engine"]),
    CategoryRule(name="compute", sku_contains=["vCPU", "RAM", "GPU", "TPU VM"]),
    # BigQuery（如果想把它单列）
    CategoryRule(name="bigquery", service_contains=["BigQuery"]),
    # VPC / Logging / Monitoring 可按需扩展
    CategoryRule(name="logging", service_contains=["Cloud Logging"]),
    CategoryRule(name="monitoring", service_contains=["Cloud Monitoring", "Operations Suite"]),
    # 兜底
    CategoryRule(name="other"),
]

def categorize(service: str, sku: str, rules: List[CategoryRule] = DEFAULT_CATEGORY_RULES) -> str:
    s = (service or "").lower()
    k = (sku or "").lower()
    for r in rules:
        hit_service = any(sub.lower() in s for sub in (r.service_contains or []))
        hit_sku     = any(sub.lower() in k for sub in (r.sku_contains or []))
        if hit_service or hit_sku or (not r.service_contains and not r.sku_contains and r.name == "other"):
            return r.name
    return "other"

# ---------------------------------------
# 2) 主函数：按天 × 类别统计净花费 (cost+credits)
# ---------------------------------------
def get_daily_billing_by_category(
    project_id: str,
    dataset_id: str,
    table_id: str,
    start_iso: str,    # 例如 "2025-08-01T00:00:00Z"
    end_iso: str,      # 例如 "2025-08-25T00:00:00Z"
    *,
    billing_project: Optional[str] = None,   # 运行查询所用的 GCP Project（通常与导出表所在项目一致）
    filter_gcp_project_id: Optional[str] = None,  # 如需仅看某个“消耗发生的 GCP 项目”，加这个过滤
    category_rules: List[CategoryRule] = DEFAULT_CATEGORY_RULES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (pivot_df, df_raw)
      - pivot_df: 行=day, 列=category, 值=net_cost
      - df_raw:   每条记录包含 day/service/sku/net_cost/category 等，便于审计
    """
    bq_project = billing_project or project_id
    client = bigquery.Client(project=bq_project)

    table_ref = f"`{project_id}.{dataset_id}.{table_id}`"

    # SQL：取日粒度 + service + sku，并把 credits（负数）并入净花费
    sql = f"""
    WITH base AS (
      SELECT
        DATE(usage_start_time) as day,
        service.description    as service,
        sku.description        as sku,
        cost,
        IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0) AS credits
      FROM {table_ref}
      WHERE usage_start_time >= @start_date
        AND usage_start_time <  @end_date
        {"AND project.id = @gcp_project_id" if filter_gcp_project_id else ""}
    )
    SELECT
      day, service, sku,
      ROUND(SUM(cost + credits), 6) AS net_cost
    FROM base
    GROUP BY day, service, sku
    ORDER BY day, service, sku
    """

    params = [
        bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_iso),
        bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_iso),
    ]
    if filter_gcp_project_id:
        params.append(bigquery.ScalarQueryParameter("gcp_project_id", "STRING", filter_gcp_project_id))

    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = job.result().to_dataframe(create_bqstorage_client=True)

    if df.empty:
        # 返回空透视表
        return pd.DataFrame(), df

    # 应用分类规则
    df["category"] = [categorize(s, k, category_rules) for s, k in zip(df["service"], df["sku"])]

    # 生成 pivot：day × category
    pivot = df.pivot_table(index="day", columns="category", values="net_cost", aggfunc="sum", fill_value=0)
    pivot = pivot.sort_index()

    return pivot, df

# -------------------------
# 3) 示例用法
# -------------------------
def main():
    PROJECT_ID  = "potent-electron-466017-q0"
    BILLING_ACCOUNT_ID = "010E46_53DFEF_91A547"
    DATASET_ID  = "billing"
    TABLE_ID    = f"gcp_billing_export_resource_v1_{BILLING_ACCOUNT_ID}"
    START_ISO   = "2025-08-01T00:00:00Z"
    END_ISO     = "2025-08-25T00:00:00Z"

    # 只看某个 GCP Project 的花费（可选）
    ONLY_GCP_PROJECT = None  # 比如 "my-prod-project"

    pivot_df, df_raw = get_daily_billing_by_category(
        project_id=PROJECT_ID,
        dataset_id=DATASET_ID,
        table_id=TABLE_ID,
        start_iso=START_ISO,
        end_iso=END_ISO,
        billing_project=PROJECT_ID,
        filter_gcp_project_id=ONLY_GCP_PROJECT,
        category_rules=DEFAULT_CATEGORY_RULES,
    )

    # 打印结果
    print("=== Daily net cost by category ===")
    print(pivot_df.tail(10))

    # 如果你想导出 CSV:
    # pivot_df.to_csv("daily_billing_by_category.csv", index=True)
    # df_raw.to_csv("daily_billing_raw_with_category.csv", index=False)
if __name__ == '__main__':
    pass