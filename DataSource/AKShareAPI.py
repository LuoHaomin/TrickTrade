import akshare as ak

# 获取基金基本信息

## 单次返回当前时刻所有历史数据
fund_name_em_df = ak.fund_name_em()
print(fund_name_em_df)

## 单次返回单只基金基本信息
fund_individual_basic_info_xq_df = ak.fund_individual_basic_info_xq(symbol="000001")
print(fund_individual_basic_info_xq_df)

## 单次返回当前时刻所有历史数据-指数基金
### symbol: choice of {"全部", "沪深指数", "行业主题", "大盘指数", "中盘指数", "小盘指数", "股票指数", "债券指数"}
### indicator: choice of {"全部", "被动指数型", "增强指数型"}
fund_info_index_em_df = ak.fund_info_index_em(symbol="沪深指数", indicator="增强指数型")
print(fund_info_index_em_df)

# 申购状态
## 单次返回当前时刻所有历史数据
fund_purchase_em_df = ak.fund_purchase_em()
print(fund_purchase_em_df)

# 基金实时行情
## ETF基金实时行情-东财：单次返回所有数据
fund_etf_spot_em_df = ak.fund_etf_spot_em()
print(fund_etf_spot_em_df)

## LOF基金实时行情-东财：单次返回所有数据
fund_lof_spot_em_df = ak.fund_lof_spot_em()
print(fund_lof_spot_em_df)

## ETF基金实时行情-同花顺：单次返回指定 date 的所有数据
fund_etf_spot_ths_df = ak.fund_etf_spot_ths(date="20240620")
print(fund_etf_spot_ths_df)

## 基金实时行情-新浪：单次返回指定 symbol 基金的所有数据
### symbol: choice of {"封闭式基金", "ETF基金", "LOF基金"}
fund_etf_category_sina_df = ak.fund_etf_category_sina(symbol="封闭式基金")
print(fund_etf_category_sina_df)

# 基金历史行情
## ETF基金历史行情-东财：单次返回指定 ETF、指定周期和指定日期间的历史行情日频率数据
### symbol: ETF 代码，ETF 代码可以在 ak.fund_etf_spot_em() 中获取
### period: choice of {'daily', 'weekly', 'monthly'}
### start_date: 开始日期，格式为 YYYYMMDD
### end_date: 结束日期，格式为 YYYYMMDD
### adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
fund_etf_hist_em_df = ak.fund_etf_hist_em(symbol="513500", period="daily", start_date="20000101", end_date="20230201", adjust="")
print(fund_etf_hist_em_df)


## LOF基金历史行情-东财：单次返回指定 LOF、指定周期和指定日期间的历史行情日频率数据
### 输入参数同上
fund_lof_hist_em_df = ak.fund_lof_hist_em(symbol="166009", period="daily", start_date="20000101", end_date="20230703", adjust="")
print(fund_lof_hist_em_df)

# 基金净值
## 开放式基金-实时数据
fund_open_fund_daily_em_df = ak.fund_open_fund_daily_em()
print(fund_open_fund_daily_em_df)

## 开放式基金-历史数据：单次返回当前时刻所有历史数据, 在查询基金数据的时候注意基金前后端问题
### symbol: 基金代码，调用 ak.fund_open_fund_daily_em() 获取
### indicator: 需要获取的指标，包括：{"单位净值走势", "累计净值走势", "累计收益率走势", "同类排名走势", "同类排名百分比", "分红送配详情", "拆分详情"}
fund_open_fund_info_em_df = ak.fund_open_fund_info_em(symbol="710001", indicator="单位净值走势")
print(fund_open_fund_info_em_df)