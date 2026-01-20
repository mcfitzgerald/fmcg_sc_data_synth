-- ============================================================================
-- Prism Consumer Goods (PCG) - ERP Schema
-- ============================================================================
-- Based on reference/fmcg_example_OLD/schema.sql
-- Target: Postgres
--
-- Structure matches the output of scripts/export_erp_format.py
-- ============================================================================

-- ============================================================================
-- DOMAIN A: SOURCE (Procurement & Inbound)
-- ============================================================================

CREATE TABLE ingredients (
    id SERIAL PRIMARY KEY,
    ingredient_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    cas_number VARCHAR(20),
    category VARCHAR(50) NOT NULL,
    purity_percent DECIMAL(5,2),
    storage_temp_min_c DECIMAL(5,2),
    storage_temp_max_c DECIMAL(5,2),
    storage_conditions VARCHAR(100),
    shelf_life_days INTEGER,
    hazmat_class VARCHAR(20),
    unit_of_measure VARCHAR(20) NOT NULL DEFAULT 'kg',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    supplier_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    tier INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    region VARCHAR(50),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    payment_terms_days INTEGER DEFAULT 30,
    currency VARCHAR(3) DEFAULT 'USD',
    qualification_status VARCHAR(20) DEFAULT 'pending',
    qualification_date DATE,
    risk_score DECIMAL(3,2),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE supplier_ingredients (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    unit_cost DECIMAL(12,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    lead_time_days INTEGER NOT NULL,
    min_order_qty DECIMAL(12,2) NOT NULL,
    is_preferred BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (supplier_id, ingredient_id)
);

CREATE TABLE certifications (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    certification_type VARCHAR(50) NOT NULL,
    issue_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    is_valid BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOMAIN B: TRANSFORM (Manufacturing)
-- ============================================================================

CREATE TABLE plants (
    id SERIAL PRIMARY KEY,
    plant_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    division_id INTEGER,
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    address VARCHAR(500),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    timezone VARCHAR(50),
    capacity_tons_per_day DECIMAL(10,2),
    operating_hours_per_day INTEGER DEFAULT 16,
    operating_days_per_week INTEGER DEFAULT 5,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE production_lines (
    id SERIAL PRIMARY KEY,
    line_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    plant_id INTEGER NOT NULL REFERENCES plants(id),
    line_type VARCHAR(50) NOT NULL,
    product_family VARCHAR(50),
    capacity_units_per_hour INTEGER NOT NULL,
    setup_time_minutes INTEGER DEFAULT 30,
    changeover_time_minutes INTEGER DEFAULT 60,
    oee_target DECIMAL(5,4) DEFAULT 0.8500,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE formulas (
    id SERIAL PRIMARY KEY,
    formula_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    product_id INTEGER,
    version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(20) DEFAULT 'draft',
    batch_size_kg DECIMAL(10,2) NOT NULL,
    yield_percent DECIMAL(5,2) DEFAULT 98.00,
    mix_time_minutes INTEGER,
    cure_time_hours INTEGER,
    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE,
    approved_by VARCHAR(100),
    approved_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE formula_ingredients (
    formula_id INTEGER NOT NULL REFERENCES formulas(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    sequence INTEGER NOT NULL,
    quantity_kg DECIMAL(10,4) NOT NULL,
    quantity_percent DECIMAL(5,2),
    is_active BOOLEAN DEFAULT true,
    tolerance_percent DECIMAL(5,2) DEFAULT 2.00,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (formula_id, ingredient_id, sequence)
);

CREATE TABLE batches (
    id SERIAL PRIMARY KEY,
    batch_number VARCHAR(30) UNIQUE NOT NULL,
    wo_id INTEGER, -- Nullable for now as we don't have WOs
    formula_id INTEGER REFERENCES formulas(id),
    plant_id INTEGER REFERENCES plants(id),
    production_line_id INTEGER REFERENCES production_lines(id),
    quantity_kg DECIMAL(12,2) NOT NULL,
    yield_percent DECIMAL(5,2),
    production_date DATE NOT NULL,
    expiry_date DATE,
    qc_status VARCHAR(20) DEFAULT 'pending',
    qc_release_date DATE,
    qc_notes TEXT,
    rejection_reason TEXT,
    data_decay_affected BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE work_orders (
    id SERIAL PRIMARY KEY,
    wo_number VARCHAR(30) UNIQUE NOT NULL,
    plant_id INTEGER NOT NULL,
    formula_id INTEGER NOT NULL,
    planned_quantity_kg DECIMAL(12,2) NOT NULL,
    planned_start_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'planned',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOMAIN C: PRODUCT (SKU Master)
-- ============================================================================

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    brand VARCHAR(50) NOT NULL,
    category VARCHAR(50) NOT NULL,
    subcategory VARCHAR(50),
    description TEXT,
    launch_date DATE,
    discontinue_date DATE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE packaging_types (
    id SERIAL PRIMARY KEY,
    packaging_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200), -- Can be NULL in export if not available
    container_type VARCHAR(50) NOT NULL,
    size_value DECIMAL(10,2) NOT NULL,
    size_unit VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE skus (
    id SERIAL PRIMARY KEY,
    sku_code VARCHAR(30) UNIQUE NOT NULL,
    name VARCHAR(300) NOT NULL,
    product_id INTEGER,
    packaging_id INTEGER REFERENCES packaging_types(id),
    formula_id INTEGER REFERENCES formulas(id),
    region VARCHAR(50),
    language VARCHAR(10),
    upc VARCHAR(20),
    ean VARCHAR(20),
    list_price DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'USD',
    shelf_life_days INTEGER,
    min_order_qty INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    launch_date DATE,
    discontinue_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sku_costs (
    id SERIAL PRIMARY KEY,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    cost_type VARCHAR(30) NOT NULL,
    cost_amount DECIMAL(10,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOMAIN D: ORDER (Demand Signal)
-- ============================================================================

CREATE TABLE channels (
    id SERIAL PRIMARY KEY,
    channel_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    channel_type VARCHAR(30) NOT NULL,
    volume_percent DECIMAL(5,2),
    margin_percent DECIMAL(5,2),
    payment_terms_days INTEGER,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE promotions (
    id SERIAL PRIMARY KEY,
    promo_code VARCHAR(30) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    promo_type VARCHAR(30),
    start_date VARCHAR(20) NOT NULL, -- Storing as string "Week X" for now
    end_date VARCHAR(20) NOT NULL,
    lift_multiplier DECIMAL(5,2),
    discount_percent DECIMAL(5,2),
    status VARCHAR(20) DEFAULT 'planned',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_number VARCHAR(30) UNIQUE NOT NULL,
    retail_account_id INTEGER,
    retail_location_id INTEGER,
    channel_id INTEGER REFERENCES channels(id),
    order_date DATE NOT NULL DEFAULT CURRENT_DATE,
    requested_delivery_date DATE,
    promised_delivery_date DATE,
    actual_delivery_date DATE,
    status VARCHAR(20) DEFAULT 'pending',
    order_type VARCHAR(20) DEFAULT 'standard',
    promo_id INTEGER REFERENCES promotions(id),
    total_cases INTEGER,
    total_amount DECIMAL(14,2),
    currency VARCHAR(3) DEFAULT 'USD',
    notes TEXT,
    is_batched BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_lines (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    line_number INTEGER NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    quantity_cases INTEGER NOT NULL,
    quantity_eaches INTEGER,
    unit_price DECIMAL(10,2) NOT NULL,
    line_amount DECIMAL(12,2),
    discount_percent DECIMAL(5,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (order_id, line_number)
);

-- ============================================================================
-- DOMAIN E: FULFILL (Outbound)
-- ============================================================================

CREATE TABLE distribution_centers (
    id SERIAL PRIMARY KEY,
    dc_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    division_id INTEGER,
    dc_type VARCHAR(30) NOT NULL,
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    address VARCHAR(500),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    capacity_cases INTEGER,
    capacity_pallets INTEGER,
    operating_hours VARCHAR(50),
    is_temperature_controlled BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE retail_locations (
    id SERIAL PRIMARY KEY,
    location_code VARCHAR(30) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    retail_account_id INTEGER,
    store_format VARCHAR(30),
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    address VARCHAR(500),
    postal_code VARCHAR(20),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    timezone VARCHAR(50),
    square_meters INTEGER,
    weekly_traffic INTEGER,
    primary_dc_id INTEGER REFERENCES distribution_centers(id),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shipments (
    id SERIAL PRIMARY KEY,
    shipment_number VARCHAR(30) UNIQUE NOT NULL,
    shipment_type VARCHAR(30),
    origin_type VARCHAR(20),
    origin_id INTEGER,
    destination_type VARCHAR(20),
    destination_id INTEGER,
    order_id INTEGER REFERENCES orders(id),
    carrier_id INTEGER,
    route_id INTEGER,
    ship_date DATE NOT NULL,
    expected_delivery_date DATE,
    actual_delivery_date DATE,
    status VARCHAR(20) DEFAULT 'planned',
    total_cases INTEGER,
    total_weight_kg DECIMAL(12,2),
    total_pallets INTEGER,
    freight_cost DECIMAL(12,2),
    currency VARCHAR(3) DEFAULT 'USD',
    tracking_number VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE shipment_lines (
    shipment_id INTEGER NOT NULL REFERENCES shipments(id),
    line_number INTEGER NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    batch_id INTEGER REFERENCES batches(id),
    quantity_cases INTEGER NOT NULL,
    quantity_eaches INTEGER,
    batch_fraction DECIMAL(5,4),
    weight_kg DECIMAL(10,2),
    lot_number VARCHAR(50),
    expiry_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (shipment_id, line_number)
);

CREATE TABLE inventory (
    id SERIAL PRIMARY KEY,
    day INTEGER,
    location_type VARCHAR(20), -- 'dc', 'store', 'plant'
    location_id INTEGER,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    batch_id INTEGER REFERENCES batches(id),
    lot_number VARCHAR(50),
    quantity_cases INTEGER NOT NULL DEFAULT 0,
    quantity_eaches INTEGER DEFAULT 0,
    quantity_reserved INTEGER DEFAULT 0,
    expiry_date DATE,
    days_until_expiry INTEGER,
    aging_bucket VARCHAR(20),
    inventory_type VARCHAR(20),
    last_movement_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_inventory_location ON inventory(location_id);
CREATE INDEX idx_skus_code ON skus(sku_code);

-- ============================================================================
-- DOMAIN E2: LOGISTICS (Transport Network)
-- ============================================================================

CREATE TABLE carriers (
    id SERIAL PRIMARY KEY,
    carrier_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    carrier_type VARCHAR(30) NOT NULL,
    sustainability_rating VARCHAR(10),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE route_segments (
    id SERIAL PRIMARY KEY,
    segment_code VARCHAR(30) UNIQUE NOT NULL,
    origin_type VARCHAR(20) NOT NULL,
    origin_id INTEGER NOT NULL,
    destination_type VARCHAR(20) NOT NULL,
    destination_id INTEGER NOT NULL,
    transport_mode VARCHAR(30) NOT NULL,
    distance_km DECIMAL(10,2),
    transit_time_hours DECIMAL(8,2),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOMAIN F: PLAN (Demand & Supply Planning)
-- ============================================================================

CREATE TABLE demand_forecasts (
    id SERIAL PRIMARY KEY,
    forecast_version VARCHAR(30) NOT NULL,
    sku_id INTEGER NOT NULL,
    location_type VARCHAR(20) NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_quantity_cases DECIMAL(12,2) NOT NULL,
    forecast_type VARCHAR(30) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOMAIN G: RETURN (Regenerate)
-- ============================================================================

CREATE TABLE returns (
    id SERIAL PRIMARY KEY,
    return_number VARCHAR(30) UNIQUE NOT NULL,
    return_date DATE NOT NULL,
    source_id INTEGER NOT NULL, -- Store
    dc_id INTEGER NOT NULL, -- DC
    status VARCHAR(20) DEFAULT 'received',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE return_lines (
    return_id INTEGER NOT NULL REFERENCES returns(id),
    line_number INTEGER NOT NULL,
    sku_id INTEGER NOT NULL,
    quantity_cases DECIMAL(12,2) NOT NULL,
    condition VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (return_id, line_number)
);

CREATE TABLE disposition_logs (
    return_id INTEGER NOT NULL REFERENCES returns(id),
    return_line_number INTEGER NOT NULL,
    disposition VARCHAR(20) NOT NULL,
    quantity_cases DECIMAL(12,2) NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- DOMAIN H: ORCHESTRATE (Hub)

CREATE TABLE kpi_thresholds (
    id SERIAL PRIMARY KEY,
    kpi_code VARCHAR(30) UNIQUE NOT NULL,
    kpi_name VARCHAR(100) NOT NULL,
    target_value DECIMAL(12,4) NOT NULL,
    warning_threshold DECIMAL(12,4),
    critical_threshold DECIMAL(12,4),
    unit VARCHAR(30) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);