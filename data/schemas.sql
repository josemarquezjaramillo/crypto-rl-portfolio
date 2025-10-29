CREATE TABLE IF NOT EXISTS public.daily_market_data
(
    id text COLLATE pg_catalog."default" NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    price numeric NOT NULL,
    market_cap numeric,
    volume numeric,
    open numeric,
    high numeric,
    low numeric,
    close numeric,
    CONSTRAINT daily_market_data_id_timestamp_key UNIQUE (id, "timestamp")
);

CREATE TABLE IF NOT EXISTS public.index_monthly_constituents
(
    period_start_date date NOT NULL,
    coin_id character varying COLLATE pg_catalog."default" NOT NULL,
    initial_market_cap_at_rebalance numeric,
    initial_weight_at_rebalance numeric,
    CONSTRAINT index_monthly_constituents_pkey PRIMARY KEY (period_start_date, coin_id)
)