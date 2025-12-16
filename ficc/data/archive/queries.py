'''
'''
standard_training_query = '''SELECT
  *
FROM
  `eng-reactor-287421.auxiliary_views_v2.trade_history_same_issue_5_yr_mat_bucket_1_materialized`
WHERE
  yield IS NOT NULL
  AND yield > 0 
  AND yield <= 3 
  AND par_traded IS NOT NULL
  AND trade_date >= '2021-07-01' 
  AND trade_date <= '2021-10-01'
  AND maturity_description_code = 2
  AND incorporated_state_code <> 'US'
  AND coupon_type = 8
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_date DESC
'''


relaxed_training_query = DATA_QUERY = '''SELECT
  *
FROM
  `eng-reactor-287421.primary_views.speedy_trade_history`
WHERE
  yield IS NOT NULL
  AND yield > 0
  AND par_traded >= 10000
  AND trade_date >= '2021-08-01'
  AND trade_date <= '2022-04-30'
  AND maturity_description_code = 2
  AND coupon_type in (8, 4, 10)
  AND capital_type <> 10
  AND default_exists <> TRUE
  AND sale_type <> 4
  AND sec_regulation IS NULL
  AND most_recent_default_event IS NULL
  AND default_indicator IS FALSE
  AND DATETIME_DIFF(trade_datetime,recent[SAFE_OFFSET(0)].trade_datetime,SECOND) < 1000000 -- 12 days to the most recent trade
  AND msrb_valid_to_date > current_date -- condition to remove cancelled trades
ORDER BY
  trade_datetime DESC 
'''
