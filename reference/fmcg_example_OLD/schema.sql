-- ============================================================================
-- Prism Consumer Goods (PCG) - FMCG Supply Chain Schema
-- ============================================================================
--
-- Spec: magical-launching-forest.md Phase 2
-- Target: ~60 tables, ~4M rows after seeding
--
-- SCOR-DS Domains:
--   A. SOURCE (Procurement & Inbound)
--   B. TRANSFORM (Manufacturing)
--   C. PRODUCT (SKU Master)
--   D. ORDER (Demand Signal)
--   E. FULFILL (Outbound)
--   E2. LOGISTICS (Transport Network)
--   E3. ESG/SUSTAINABILITY
--   F. PLAN (Demand & Supply Planning)
--   G. RETURN (Regenerate)
--   H. ORCHESTRATE (Hub)
--
-- ============================================================================

-- ============================================================================
-- DOMAIN A: SOURCE (Procurement & Inbound) - SCOR: Source
-- ============================================================================
-- The procurement cycle: suppliers → ingredients → POs → goods receipts

-- A1: ingredients - Raw chemicals with CAS numbers, purity, storage
CREATE TABLE ingredients (
    id SERIAL PRIMARY KEY,
    ingredient_code VARCHAR(20) UNIQUE NOT NULL,  -- e.g., ING-PALM-001, ING-SORB-001
    name VARCHAR(200) NOT NULL,
    cas_number VARCHAR(20),                        -- Chemical Abstracts Service number
    category VARCHAR(50) NOT NULL,                 -- surfactant, abrasive, humectant, flavor, etc.
    purity_percent DECIMAL(5,2),                   -- 99.50 = 99.5% pure
    storage_temp_min_c DECIMAL(5,2),
    storage_temp_max_c DECIMAL(5,2),
    storage_conditions VARCHAR(100),               -- ambient, refrigerated, frozen, humidity-controlled
    shelf_life_days INTEGER,
    hazmat_class VARCHAR(20),                      -- flammable, corrosive, oxidizer, none
    unit_of_measure VARCHAR(20) NOT NULL DEFAULT 'kg',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ingredients_category ON ingredients(category);
CREATE INDEX idx_ingredients_hazmat ON ingredients(hazmat_class);
CREATE INDEX idx_ingredients_active ON ingredients(is_active) WHERE is_active = true;

-- A2: suppliers - Global supplier network (Tier 1/2/3)
CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    supplier_code VARCHAR(20) UNIQUE NOT NULL,     -- e.g., SUP-PALM-MY-001
    name VARCHAR(200) NOT NULL,
    tier INTEGER NOT NULL CHECK (tier IN (1, 2, 3)),
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    region VARCHAR(50),                            -- APAC, EUR, NAM, LATAM, AFR-EUR
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    payment_terms_days INTEGER DEFAULT 30,
    currency VARCHAR(3) DEFAULT 'USD',
    qualification_status VARCHAR(20) DEFAULT 'pending'
        CHECK (qualification_status IN ('pending', 'qualified', 'probation', 'disqualified')),
    qualification_date DATE,
    risk_score DECIMAL(3,2),                       -- 0.00 to 1.00 (higher = more risk)
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_suppliers_tier ON suppliers(tier);
CREATE INDEX idx_suppliers_country ON suppliers(country);
CREATE INDEX idx_suppliers_region ON suppliers(region);
CREATE INDEX idx_suppliers_qualification ON suppliers(qualification_status);
CREATE INDEX idx_suppliers_active ON suppliers(is_active) WHERE is_active = true;

-- A3: supplier_ingredients - M:M with lead times, MOQs, costs
CREATE TABLE supplier_ingredients (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    supplier_part_number VARCHAR(50),
    unit_cost DECIMAL(12,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    lead_time_days INTEGER NOT NULL,
    min_order_qty DECIMAL(12,2) NOT NULL,          -- MOQ in ingredient's UoM
    order_multiple DECIMAL(12,2) DEFAULT 1,        -- Must order in multiples of this
    is_preferred BOOLEAN DEFAULT false,
    is_approved BOOLEAN DEFAULT true,
    approval_date DATE,
    contract_start_date DATE,
    contract_end_date DATE,
    on_time_delivery_rate DECIMAL(5,4),            -- 0.9500 = 95%
    quality_acceptance_rate DECIMAL(5,4),          -- 0.9900 = 99%
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (supplier_id, ingredient_id)
);

CREATE INDEX idx_supplier_ingredients_supplier ON supplier_ingredients(supplier_id);
CREATE INDEX idx_supplier_ingredients_ingredient ON supplier_ingredients(ingredient_id);
CREATE INDEX idx_supplier_ingredients_preferred ON supplier_ingredients(is_preferred) WHERE is_preferred = true;
CREATE INDEX idx_supplier_ingredients_contract ON supplier_ingredients(contract_start_date, contract_end_date);

-- A4: certifications - ISO, GMP, Halal, Kosher, RSPO
CREATE TABLE certifications (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    certification_type VARCHAR(50) NOT NULL,       -- ISO9001, ISO14001, GMP, Halal, Kosher, RSPO, FSC
    certification_body VARCHAR(200),               -- Certifying organization
    certificate_number VARCHAR(100),
    issue_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    scope VARCHAR(500),                            -- What's covered by the certification
    is_valid BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_certifications_supplier ON certifications(supplier_id);
CREATE INDEX idx_certifications_type ON certifications(certification_type);
CREATE INDEX idx_certifications_expiry ON certifications(expiry_date);
CREATE INDEX idx_certifications_valid ON certifications(is_valid) WHERE is_valid = true;

-- A5: purchase_orders - POs to suppliers
CREATE TABLE purchase_orders (
    id SERIAL PRIMARY KEY,
    po_number VARCHAR(30) UNIQUE NOT NULL,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    plant_id INTEGER,                              -- FK added after plants table created
    order_date DATE NOT NULL DEFAULT CURRENT_DATE,
    requested_date DATE NOT NULL,
    promised_date DATE,
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'submitted', 'confirmed', 'shipped', 'partial', 'received', 'cancelled')),
    total_amount DECIMAL(14,2),
    currency VARCHAR(3) DEFAULT 'USD',
    incoterms VARCHAR(10),                         -- FOB, CIF, DDP, etc.
    notes TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_purchase_orders_supplier ON purchase_orders(supplier_id);
CREATE INDEX idx_purchase_orders_plant ON purchase_orders(plant_id);
CREATE INDEX idx_purchase_orders_status ON purchase_orders(status);
CREATE INDEX idx_purchase_orders_date ON purchase_orders(order_date);
CREATE INDEX idx_purchase_orders_requested ON purchase_orders(requested_date);

-- A6: purchase_order_lines - PO line items
CREATE TABLE purchase_order_lines (
    po_id INTEGER NOT NULL REFERENCES purchase_orders(id),
    line_number INTEGER NOT NULL,
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    quantity DECIMAL(12,2) NOT NULL,
    unit_of_measure VARCHAR(20) NOT NULL DEFAULT 'kg',
    unit_price DECIMAL(12,4) NOT NULL,
    line_amount DECIMAL(14,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    requested_date DATE,
    status VARCHAR(20) DEFAULT 'open'
        CHECK (status IN ('open', 'partial', 'received', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (po_id, line_number)
);

CREATE INDEX idx_po_lines_ingredient ON purchase_order_lines(ingredient_id);
CREATE INDEX idx_po_lines_status ON purchase_order_lines(status);

-- A7: goods_receipts - Receipt of goods from suppliers
CREATE TABLE goods_receipts (
    id SERIAL PRIMARY KEY,
    gr_number VARCHAR(30) UNIQUE NOT NULL,
    po_id INTEGER NOT NULL REFERENCES purchase_orders(id),
    plant_id INTEGER,                              -- FK added after plants table created
    receipt_date DATE NOT NULL DEFAULT CURRENT_DATE,
    received_by VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'inspecting', 'accepted', 'rejected', 'partial')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_goods_receipts_po ON goods_receipts(po_id);
CREATE INDEX idx_goods_receipts_plant ON goods_receipts(plant_id);
CREATE INDEX idx_goods_receipts_date ON goods_receipts(receipt_date);
CREATE INDEX idx_goods_receipts_status ON goods_receipts(status);

-- A8: goods_receipt_lines - Actual received qty, lot numbers, quality status
CREATE TABLE goods_receipt_lines (
    gr_id INTEGER NOT NULL REFERENCES goods_receipts(id),
    line_number INTEGER NOT NULL,
    po_id INTEGER NOT NULL,
    po_line_number INTEGER NOT NULL,
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    quantity_received DECIMAL(12,2) NOT NULL,
    unit_of_measure VARCHAR(20) NOT NULL DEFAULT 'kg',
    lot_number VARCHAR(50) NOT NULL,               -- Supplier's lot number
    manufacture_date DATE,
    expiry_date DATE,
    quality_status VARCHAR(20) DEFAULT 'pending'
        CHECK (quality_status IN ('pending', 'approved', 'rejected', 'hold')),
    quality_notes TEXT,
    storage_location VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (gr_id, line_number),
    FOREIGN KEY (po_id, po_line_number) REFERENCES purchase_order_lines(po_id, line_number)
);

CREATE INDEX idx_gr_lines_ingredient ON goods_receipt_lines(ingredient_id);
CREATE INDEX idx_gr_lines_lot ON goods_receipt_lines(lot_number);
CREATE INDEX idx_gr_lines_quality ON goods_receipt_lines(quality_status);
CREATE INDEX idx_gr_lines_expiry ON goods_receipt_lines(expiry_date);

-- ============================================================================
-- DOMAIN B: TRANSFORM (Manufacturing) - SCOR: Transform
-- ============================================================================
-- Manufacturing: plants → production lines → formulas → work orders → batches

-- B1: plants - 7 manufacturing facilities globally
CREATE TABLE plants (
    id SERIAL PRIMARY KEY,
    plant_code VARCHAR(20) UNIQUE NOT NULL,        -- e.g., PLT-NAM-TN, PLT-APAC-CN
    name VARCHAR(200) NOT NULL,
    division_id INTEGER,                           -- FK added after divisions table created
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

CREATE INDEX idx_plants_division ON plants(division_id);
CREATE INDEX idx_plants_country ON plants(country);
CREATE INDEX idx_plants_active ON plants(is_active) WHERE is_active = true;

-- B2: production_lines - Manufacturing line capacity
CREATE TABLE production_lines (
    id SERIAL PRIMARY KEY,
    line_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    plant_id INTEGER NOT NULL REFERENCES plants(id),
    line_type VARCHAR(50) NOT NULL,                -- mixing, filling, packaging, labeling
    product_family VARCHAR(50),                    -- oral_care, home_care, personal_care, or NULL for flexible
    capacity_units_per_hour INTEGER NOT NULL,
    setup_time_minutes INTEGER DEFAULT 30,
    changeover_time_minutes INTEGER DEFAULT 60,
    oee_target DECIMAL(5,4) DEFAULT 0.8500,        -- Overall Equipment Effectiveness target
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_production_lines_plant ON production_lines(plant_id);
CREATE INDEX idx_production_lines_type ON production_lines(line_type);
CREATE INDEX idx_production_lines_family ON production_lines(product_family);
CREATE INDEX idx_production_lines_active ON production_lines(is_active) WHERE is_active = true;

-- B3: formulas - Recipe/BOM definitions with yield parameters
CREATE TABLE formulas (
    id SERIAL PRIMARY KEY,
    formula_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    product_id INTEGER,                            -- FK added after products table created
    version INTEGER NOT NULL DEFAULT 1,
    status VARCHAR(20) DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'obsolete')),
    batch_size_kg DECIMAL(10,2) NOT NULL,          -- Standard batch size
    yield_percent DECIMAL(5,2) DEFAULT 98.00,      -- Expected yield
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

CREATE INDEX idx_formulas_product ON formulas(product_id);
CREATE INDEX idx_formulas_status ON formulas(status);
CREATE INDEX idx_formulas_effective ON formulas(effective_from, effective_to);

-- B4: formula_ingredients - Composition with sequence, quantities (composite PK)
CREATE TABLE formula_ingredients (
    formula_id INTEGER NOT NULL REFERENCES formulas(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    sequence INTEGER NOT NULL,                     -- Processing order
    quantity_kg DECIMAL(10,4) NOT NULL,            -- Amount per standard batch
    quantity_percent DECIMAL(5,2),                 -- Percentage of total batch
    is_active BOOLEAN DEFAULT true,                -- For phase-in/phase-out
    tolerance_percent DECIMAL(5,2) DEFAULT 2.00,   -- Acceptable variance
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (formula_id, ingredient_id, sequence)
);

CREATE INDEX idx_formula_ingredients_ingredient ON formula_ingredients(ingredient_id);

-- B5: work_orders - Scheduled production (links Plan to Transform)
CREATE TABLE work_orders (
    id SERIAL PRIMARY KEY,
    wo_number VARCHAR(30) UNIQUE NOT NULL,
    formula_id INTEGER NOT NULL REFERENCES formulas(id),
    plant_id INTEGER NOT NULL REFERENCES plants(id),
    production_line_id INTEGER REFERENCES production_lines(id),
    planned_quantity_kg DECIMAL(12,2) NOT NULL,
    actual_quantity_kg DECIMAL(12,2),
    planned_start_date DATE NOT NULL,
    planned_end_date DATE NOT NULL,
    actual_start_datetime TIMESTAMP,
    actual_end_datetime TIMESTAMP,
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'released', 'in_progress', 'completed', 'cancelled', 'on_hold')),
    priority INTEGER DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    notes TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_work_orders_formula ON work_orders(formula_id);
CREATE INDEX idx_work_orders_plant ON work_orders(plant_id);
CREATE INDEX idx_work_orders_line ON work_orders(production_line_id);
CREATE INDEX idx_work_orders_status ON work_orders(status);
CREATE INDEX idx_work_orders_dates ON work_orders(planned_start_date, planned_end_date);
CREATE INDEX idx_work_orders_priority ON work_orders(priority);

-- B6: work_order_materials - Planned material consumption per work order
CREATE TABLE work_order_materials (
    wo_id INTEGER NOT NULL REFERENCES work_orders(id),
    line_number INTEGER NOT NULL,
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    planned_quantity_kg DECIMAL(12,4) NOT NULL,
    actual_quantity_kg DECIMAL(12,4),
    lot_number VARCHAR(50),                        -- Assigned lot from inventory
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'issued', 'consumed', 'returned')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (wo_id, line_number)
);

CREATE INDEX idx_wo_materials_ingredient ON work_order_materials(ingredient_id);
CREATE INDEX idx_wo_materials_lot ON work_order_materials(lot_number);

-- B7: batches - Production lots with QC status, expiry
CREATE TABLE batches (
    id SERIAL PRIMARY KEY,
    batch_number VARCHAR(30) UNIQUE NOT NULL,      -- e.g., B-2024-RECALL-001
    wo_id INTEGER NOT NULL REFERENCES work_orders(id),
    formula_id INTEGER NOT NULL REFERENCES formulas(id),
    plant_id INTEGER NOT NULL REFERENCES plants(id),
    production_line_id INTEGER REFERENCES production_lines(id),
    quantity_kg DECIMAL(12,2) NOT NULL,
    yield_percent DECIMAL(5,2),
    production_date DATE NOT NULL,
    expiry_date DATE NOT NULL,
    qc_status VARCHAR(20) DEFAULT 'pending'
        CHECK (qc_status IN ('pending', 'approved', 'rejected', 'hold', 'quarantine')),
    qc_release_date DATE,
    qc_notes TEXT,
    rejection_reason TEXT,                         -- e.g., quality_decay, random_sample
    data_decay_affected BOOLEAN DEFAULT false,     -- For validation tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_batches_wo ON batches(wo_id);
CREATE INDEX idx_batches_formula ON batches(formula_id);
CREATE INDEX idx_batches_plant ON batches(plant_id);
CREATE INDEX idx_batches_production_date ON batches(production_date);
CREATE INDEX idx_batches_expiry ON batches(expiry_date);
CREATE INDEX idx_batches_qc_status ON batches(qc_status);
CREATE INDEX idx_batches_decay ON batches(data_decay_affected) WHERE data_decay_affected = true;

-- B8: batch_ingredients - Actual consumption for mass balance
CREATE TABLE batch_ingredients (
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
    sequence INTEGER NOT NULL,
    planned_quantity_kg DECIMAL(12,4) NOT NULL,
    actual_quantity_kg DECIMAL(12,4) NOT NULL,
    lot_number VARCHAR(50) NOT NULL,               -- Source lot from goods receipt
    variance_kg DECIMAL(12,4) GENERATED ALWAYS AS (actual_quantity_kg - planned_quantity_kg) STORED,
    variance_percent DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (batch_id, ingredient_id, sequence)
);

CREATE INDEX idx_batch_ingredients_ingredient ON batch_ingredients(ingredient_id);
CREATE INDEX idx_batch_ingredients_lot ON batch_ingredients(lot_number);

-- B9: batch_cost_ledger - Material, labor, energy, overhead costs
CREATE TABLE batch_cost_ledger (
    id SERIAL PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    cost_type VARCHAR(30) NOT NULL
        CHECK (cost_type IN ('material', 'labor', 'energy', 'overhead', 'scrap', 'rework')),
    cost_amount DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    cost_driver VARCHAR(100),                      -- What caused this cost
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_batch_cost_batch ON batch_cost_ledger(batch_id);
CREATE INDEX idx_batch_cost_type ON batch_cost_ledger(cost_type);

-- ============================================================================
-- DOMAIN C: PRODUCT (SKU Master) - Shared across ORDER/FULFILL
-- ============================================================================
-- Product hierarchy: products → packaging_types → SKUs

-- C1: products - Product families (PrismWhite, ClearWave, AquaPure)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    brand VARCHAR(50) NOT NULL,                    -- PrismWhite, ClearWave, AquaPure
    category VARCHAR(50) NOT NULL,                 -- oral_care, home_care, personal_care
    subcategory VARCHAR(50),                       -- toothpaste, dish_soap, body_wash
    description TEXT,
    launch_date DATE,
    discontinue_date DATE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = true;

-- C2: packaging_types - Tubes, bottles, sizes, regional variants
CREATE TABLE packaging_types (
    id SERIAL PRIMARY KEY,
    packaging_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    container_type VARCHAR(50) NOT NULL,           -- tube, bottle, pouch, jar, box
    size_value DECIMAL(10,2) NOT NULL,
    size_unit VARCHAR(20) NOT NULL,                -- ml, oz, g, count
    material VARCHAR(50),                          -- plastic, aluminum, glass, paper
    is_recyclable BOOLEAN DEFAULT false,
    units_per_case INTEGER NOT NULL DEFAULT 12,
    case_weight_kg DECIMAL(10,3),
    case_dimensions_cm VARCHAR(50),                -- LxWxH
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_packaging_container ON packaging_types(container_type);
CREATE INDEX idx_packaging_recyclable ON packaging_types(is_recyclable);

-- C3: skus - The explosion point (~2,000 SKUs)
CREATE TABLE skus (
    id SERIAL PRIMARY KEY,
    sku_code VARCHAR(30) UNIQUE NOT NULL,          -- e.g., SKU-PW-TP-6OZ-US
    name VARCHAR(300) NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    packaging_id INTEGER NOT NULL REFERENCES packaging_types(id),
    formula_id INTEGER REFERENCES formulas(id),
    region VARCHAR(50),                            -- NAM, LATAM, APAC, EUR, AFR-EUR, GLOBAL
    language VARCHAR(10),                          -- en, es, pt, zh, hi, fr, ar
    upc VARCHAR(20),                               -- Universal Product Code
    ean VARCHAR(20),                               -- European Article Number
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

CREATE INDEX idx_skus_product ON skus(product_id);
CREATE INDEX idx_skus_packaging ON skus(packaging_id);
CREATE INDEX idx_skus_formula ON skus(formula_id);
CREATE INDEX idx_skus_region ON skus(region);
CREATE INDEX idx_skus_active ON skus(is_active) WHERE is_active = true;
CREATE INDEX idx_skus_upc ON skus(upc);

-- C4: sku_costs - Standard costs by type
CREATE TABLE sku_costs (
    id SERIAL PRIMARY KEY,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    cost_type VARCHAR(30) NOT NULL
        CHECK (cost_type IN ('material', 'labor', 'overhead', 'packaging', 'freight', 'landed')),
    cost_amount DECIMAL(10,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (sku_id, cost_type, effective_from)
);

CREATE INDEX idx_sku_costs_sku ON sku_costs(sku_id);
CREATE INDEX idx_sku_costs_type ON sku_costs(cost_type);
CREATE INDEX idx_sku_costs_effective ON sku_costs(effective_from, effective_to);

-- C5: sku_substitutes - Substitute/equivalent SKUs (symmetric relationship)
CREATE TABLE sku_substitutes (
    id SERIAL PRIMARY KEY,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    substitute_sku_id INTEGER NOT NULL REFERENCES skus(id),
    substitute_type VARCHAR(30) NOT NULL
        CHECK (substitute_type IN ('size_variant', 'regional_variant', 'promotional', 'emergency')),
    priority INTEGER DEFAULT 1,                    -- Lower = preferred substitute
    conversion_factor DECIMAL(8,4) DEFAULT 1.0,    -- For size variants: 1.5 = 1.5x the substitute
    is_bidirectional BOOLEAN DEFAULT true,         -- If true, relationship goes both ways
    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (sku_id != substitute_sku_id)
);

CREATE INDEX idx_sku_substitutes_sku ON sku_substitutes(sku_id);
CREATE INDEX idx_sku_substitutes_substitute ON sku_substitutes(substitute_sku_id);
CREATE INDEX idx_sku_substitutes_type ON sku_substitutes(substitute_type);

-- ============================================================================
-- DOMAIN D: ORDER (Demand Signal) - SCOR: Order
-- ============================================================================
-- Demand capture: channels → promotions → orders → allocations

-- D1: channels - 4 channel types
CREATE TABLE channels (
    id SERIAL PRIMARY KEY,
    channel_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    channel_type VARCHAR(30) NOT NULL
        CHECK (channel_type IN ('bm_large', 'bm_distributor', 'ecommerce', 'dtc')),
    volume_percent DECIMAL(5,2),                   -- Percentage of total volume
    margin_percent DECIMAL(5,2),                   -- Typical margin
    payment_terms_days INTEGER,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_channels_type ON channels(channel_type);
CREATE INDEX idx_channels_active ON channels(is_active) WHERE is_active = true;

-- D2: promotions - Trade promos with lift multipliers and hangover effects
CREATE TABLE promotions (
    id SERIAL PRIMARY KEY,
    promo_code VARCHAR(30) UNIQUE NOT NULL,        -- e.g., PROMO-BF-2024
    name VARCHAR(200) NOT NULL,
    promo_type VARCHAR(30) NOT NULL
        CHECK (promo_type IN ('price_discount', 'bogo', 'display', 'feature', 'tpr', 'seasonal')),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    lift_multiplier DECIMAL(5,2) NOT NULL,         -- 2.50 = 150% increase
    hangover_multiplier DECIMAL(5,2) DEFAULT 0.70, -- 0.70 = 30% decrease post-promo
    hangover_weeks INTEGER DEFAULT 1,              -- How many weeks of hangover
    discount_percent DECIMAL(5,2),
    trade_spend_budget DECIMAL(14,2),
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'active', 'completed', 'cancelled')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_promotions_dates ON promotions(start_date, end_date);
CREATE INDEX idx_promotions_status ON promotions(status);
CREATE INDEX idx_promotions_type ON promotions(promo_type);

-- D2b: promotion_skus - Which SKUs are included in each promotion
CREATE TABLE promotion_skus (
    promo_id INTEGER NOT NULL REFERENCES promotions(id),
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    specific_discount_percent DECIMAL(5,2),        -- Override promo-level discount
    specific_lift_multiplier DECIMAL(5,2),         -- Override promo-level lift
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (promo_id, sku_id)
);

CREATE INDEX idx_promo_skus_sku ON promotion_skus(sku_id);

-- D2c: promotion_accounts - Which accounts run each promotion
CREATE TABLE promotion_accounts (
    promo_id INTEGER NOT NULL REFERENCES promotions(id),
    retail_account_id INTEGER,                     -- FK added after retail_accounts created
    trade_spend_allocation DECIMAL(14,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (promo_id, retail_account_id)
);

-- D3: orders - Customer orders (~200K)
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_number VARCHAR(30) UNIQUE NOT NULL,
    retail_account_id INTEGER,                     -- FK added after retail_accounts created
    retail_location_id INTEGER,                    -- FK added after retail_locations created
    channel_id INTEGER NOT NULL REFERENCES channels(id),
    order_date DATE NOT NULL DEFAULT CURRENT_DATE,
    requested_delivery_date DATE NOT NULL,
    promised_delivery_date DATE,
    actual_delivery_date DATE,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'confirmed', 'allocated', 'picking', 'shipped', 'delivered', 'cancelled')),
    order_type VARCHAR(20) DEFAULT 'standard'
        CHECK (order_type IN ('standard', 'rush', 'backorder', 'promotional')),
    promo_id INTEGER REFERENCES promotions(id),
    total_cases INTEGER,
    total_amount DECIMAL(14,2),
    currency VARCHAR(3) DEFAULT 'USD',
    notes TEXT,
    is_batched BOOLEAN DEFAULT false,              -- For bullwhip quirk tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_orders_account ON orders(retail_account_id);
CREATE INDEX idx_orders_location ON orders(retail_location_id);
CREATE INDEX idx_orders_channel ON orders(channel_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_delivery ON orders(requested_delivery_date);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_promo ON orders(promo_id);

-- D4: order_lines - Line items with SKU, qty, price (composite PK)
CREATE TABLE order_lines (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    line_number INTEGER NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    quantity_cases INTEGER NOT NULL,
    quantity_eaches INTEGER GENERATED ALWAYS AS (quantity_cases * 12) STORED,  -- Assume 12 per case
    unit_price DECIMAL(10,2) NOT NULL,
    line_amount DECIMAL(12,2) GENERATED ALWAYS AS (quantity_cases * unit_price) STORED,
    discount_percent DECIMAL(5,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open'
        CHECK (status IN ('open', 'allocated', 'partial', 'shipped', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (order_id, line_number)
);

CREATE INDEX idx_order_lines_sku ON order_lines(sku_id);
CREATE INDEX idx_order_lines_status ON order_lines(status);

-- D5: order_allocations - ATP/allocation of inventory to orders
CREATE TABLE order_allocations (
    id SERIAL PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    order_line_number INTEGER NOT NULL,
    dc_id INTEGER,                                 -- FK added after distribution_centers created
    batch_id INTEGER REFERENCES batches(id),
    allocated_cases INTEGER NOT NULL,
    allocation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiry_date TIMESTAMP,                         -- Allocation expires if not fulfilled
    status VARCHAR(20) DEFAULT 'allocated'
        CHECK (status IN ('allocated', 'picked', 'shipped', 'cancelled', 'expired')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id, order_line_number) REFERENCES order_lines(order_id, line_number)
);

CREATE INDEX idx_order_allocations_order ON order_allocations(order_id);
CREATE INDEX idx_order_allocations_dc ON order_allocations(dc_id);
CREATE INDEX idx_order_allocations_batch ON order_allocations(batch_id);
CREATE INDEX idx_order_allocations_status ON order_allocations(status);

-- ============================================================================
-- DOMAIN E: FULFILL (Outbound) - SCOR: Fulfill
-- ============================================================================
-- Distribution network: divisions → DCs → ports → retail accounts → locations

-- E1: divisions - 5 global divisions
CREATE TABLE divisions (
    id SERIAL PRIMARY KEY,
    division_code VARCHAR(20) UNIQUE NOT NULL,     -- NAM, LATAM, APAC, EUR, AFR-EUR
    name VARCHAR(100) NOT NULL,
    headquarters_city VARCHAR(100) NOT NULL,
    headquarters_country VARCHAR(100) NOT NULL,
    president VARCHAR(200),
    revenue_target DECIMAL(14,2),
    currency VARCHAR(3) DEFAULT 'USD',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_divisions_active ON divisions(is_active) WHERE is_active = true;

-- E2: distribution_centers - ~25 DCs globally
CREATE TABLE distribution_centers (
    id SERIAL PRIMARY KEY,
    dc_code VARCHAR(20) UNIQUE NOT NULL,           -- e.g., DC-NAM-CHI-001
    name VARCHAR(200) NOT NULL,
    division_id INTEGER NOT NULL REFERENCES divisions(id),
    dc_type VARCHAR(30) NOT NULL
        CHECK (dc_type IN ('regional', 'national', 'cross_dock', 'ecommerce')),
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    address VARCHAR(500),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    capacity_cases INTEGER,
    capacity_pallets INTEGER,
    operating_hours VARCHAR(50),                   -- e.g., "06:00-22:00"
    is_temperature_controlled BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_dcs_division ON distribution_centers(division_id);
CREATE INDEX idx_dcs_type ON distribution_centers(dc_type);
CREATE INDEX idx_dcs_country ON distribution_centers(country);
CREATE INDEX idx_dcs_active ON distribution_centers(is_active) WHERE is_active = true;

-- E3: ports - Ocean/air freight nodes for multi-leg routing
CREATE TABLE ports (
    id SERIAL PRIMARY KEY,
    port_code VARCHAR(20) UNIQUE NOT NULL,         -- UN/LOCODE e.g., USCHI, CNSHA
    name VARCHAR(200) NOT NULL,
    port_type VARCHAR(30) NOT NULL
        CHECK (port_type IN ('ocean', 'air', 'rail', 'multimodal')),
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    timezone VARCHAR(50),
    handling_capacity_teu INTEGER,                 -- TEU for ocean ports
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ports_type ON ports(port_type);
CREATE INDEX idx_ports_country ON ports(country);
CREATE INDEX idx_ports_active ON ports(is_active) WHERE is_active = true;

-- E4: retail_accounts - Archetype-based accounts (~100)
CREATE TABLE retail_accounts (
    id SERIAL PRIMARY KEY,
    account_code VARCHAR(20) UNIQUE NOT NULL,      -- e.g., ACCT-MEGA-001
    name VARCHAR(200) NOT NULL,
    account_type VARCHAR(30) NOT NULL
        CHECK (account_type IN ('megamart', 'valueclub', 'urbanessential', 'regional_grocer', 'indie_retail', 'digital_first', 'omni_retailer', 'prism_direct')),
    channel_id INTEGER NOT NULL REFERENCES channels(id),
    division_id INTEGER NOT NULL REFERENCES divisions(id),
    parent_account_id INTEGER REFERENCES retail_accounts(id),
    headquarters_country VARCHAR(100),
    headquarters_city VARCHAR(100),
    store_count INTEGER,                           -- Number of locations
    annual_volume_cases INTEGER,
    payment_terms_days INTEGER DEFAULT 30,
    credit_limit DECIMAL(14,2),
    is_strategic BOOLEAN DEFAULT false,            -- Key account flag
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_retail_accounts_type ON retail_accounts(account_type);
CREATE INDEX idx_retail_accounts_channel ON retail_accounts(channel_id);
CREATE INDEX idx_retail_accounts_division ON retail_accounts(division_id);
CREATE INDEX idx_retail_accounts_parent ON retail_accounts(parent_account_id);
CREATE INDEX idx_retail_accounts_strategic ON retail_accounts(is_strategic) WHERE is_strategic = true;
CREATE INDEX idx_retail_accounts_active ON retail_accounts(is_active) WHERE is_active = true;

-- E5: retail_locations - Individual stores (~10,000)
CREATE TABLE retail_locations (
    id SERIAL PRIMARY KEY,
    location_code VARCHAR(30) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    retail_account_id INTEGER NOT NULL REFERENCES retail_accounts(id),
    store_format VARCHAR(30),                      -- hypermarket, supermarket, convenience, pharmacy
    country VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    address VARCHAR(500),
    postal_code VARCHAR(20),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    timezone VARCHAR(50),
    square_meters INTEGER,
    weekly_traffic INTEGER,                        -- Foot traffic
    primary_dc_id INTEGER REFERENCES distribution_centers(id),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_retail_locations_account ON retail_locations(retail_account_id);
CREATE INDEX idx_retail_locations_format ON retail_locations(store_format);
CREATE INDEX idx_retail_locations_country ON retail_locations(country);
CREATE INDEX idx_retail_locations_dc ON retail_locations(primary_dc_id);
CREATE INDEX idx_retail_locations_active ON retail_locations(is_active) WHERE is_active = true;

-- E6: shipments - Physical movements (~180K)
CREATE TABLE shipments (
    id SERIAL PRIMARY KEY,
    shipment_number VARCHAR(30) UNIQUE NOT NULL,
    shipment_type VARCHAR(30) NOT NULL
        CHECK (shipment_type IN ('plant_to_dc', 'dc_to_dc', 'dc_to_store', 'direct_to_store', 'return')),
    origin_type VARCHAR(20) NOT NULL
        CHECK (origin_type IN ('plant', 'dc', 'port', 'store')),
    origin_id INTEGER NOT NULL,                    -- Polymorphic FK
    destination_type VARCHAR(20) NOT NULL
        CHECK (destination_type IN ('dc', 'port', 'store')),
    destination_id INTEGER NOT NULL,               -- Polymorphic FK
    order_id INTEGER REFERENCES orders(id),
    carrier_id INTEGER,                            -- FK added after carriers created
    route_id INTEGER,                              -- FK added after routes created
    ship_date DATE NOT NULL,
    expected_delivery_date DATE NOT NULL,
    actual_delivery_date DATE,
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'loading', 'in_transit', 'at_port', 'delivered', 'exception')),
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

CREATE INDEX idx_shipments_type ON shipments(shipment_type);
CREATE INDEX idx_shipments_origin ON shipments(origin_type, origin_id);
CREATE INDEX idx_shipments_destination ON shipments(destination_type, destination_id);
CREATE INDEX idx_shipments_order ON shipments(order_id);
CREATE INDEX idx_shipments_carrier ON shipments(carrier_id);
CREATE INDEX idx_shipments_route ON shipments(route_id);
CREATE INDEX idx_shipments_ship_date ON shipments(ship_date);
CREATE INDEX idx_shipments_status ON shipments(status);

-- E7: shipment_lines - Items with batch numbers for lot tracking (~540K)
CREATE TABLE shipment_lines (
    shipment_id INTEGER NOT NULL REFERENCES shipments(id),
    line_number INTEGER NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    quantity_cases INTEGER NOT NULL,
    quantity_eaches INTEGER,
    batch_fraction DECIMAL(5,4),                   -- Fraction of batch (for splitting)
    weight_kg DECIMAL(10,2),
    lot_number VARCHAR(50) NOT NULL,               -- For traceability
    expiry_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (shipment_id, line_number)
);

CREATE INDEX idx_shipment_lines_sku ON shipment_lines(sku_id);
CREATE INDEX idx_shipment_lines_batch ON shipment_lines(batch_id);
CREATE INDEX idx_shipment_lines_lot ON shipment_lines(lot_number);
CREATE INDEX idx_shipment_lines_expiry ON shipment_lines(expiry_date);

-- E8: inventory - Stock by location with aging buckets and inventory waterfall
CREATE TABLE inventory (
    id SERIAL PRIMARY KEY,
    location_type VARCHAR(20) NOT NULL
        CHECK (location_type IN ('plant', 'dc', 'store', 'in_transit')),
    location_id INTEGER NOT NULL,                  -- Polymorphic FK
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    batch_id INTEGER REFERENCES batches(id),
    lot_number VARCHAR(50),
    quantity_cases INTEGER NOT NULL DEFAULT 0,
    quantity_eaches INTEGER DEFAULT 0,
    quantity_reserved INTEGER DEFAULT 0,           -- Allocated but not shipped
    quantity_available INTEGER GENERATED ALWAYS AS (quantity_cases - quantity_reserved) STORED,
    expiry_date DATE,
    days_until_expiry INTEGER,
    aging_bucket VARCHAR(20)                       -- 0-30, 31-60, 61-90, 90+
        CHECK (aging_bucket IN ('0-30', '31-60', '61-90', '90+')),
    inventory_type VARCHAR(20)                     -- safety_stock, cycle_stock (for waterfall view)
        CHECK (inventory_type IS NULL OR inventory_type IN ('safety_stock', 'cycle_stock')),
    last_movement_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (location_type, location_id, sku_id, batch_id)
);

CREATE INDEX idx_inventory_location ON inventory(location_type, location_id);
CREATE INDEX idx_inventory_sku ON inventory(sku_id);
CREATE INDEX idx_inventory_batch ON inventory(batch_id);
CREATE INDEX idx_inventory_expiry ON inventory(expiry_date);
CREATE INDEX idx_inventory_aging ON inventory(aging_bucket);
CREATE INDEX idx_inventory_type ON inventory(inventory_type);

-- E9: pick_waves - Picking/packing execution
CREATE TABLE pick_waves (
    id SERIAL PRIMARY KEY,
    wave_number VARCHAR(30) UNIQUE NOT NULL,
    dc_id INTEGER NOT NULL REFERENCES distribution_centers(id),
    wave_date DATE NOT NULL DEFAULT CURRENT_DATE,
    wave_type VARCHAR(20) DEFAULT 'standard'
        CHECK (wave_type IN ('standard', 'rush', 'replenishment')),
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'released', 'picking', 'packing', 'staged', 'loaded', 'completed')),
    total_orders INTEGER,
    total_lines INTEGER,
    total_cases INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pick_waves_dc ON pick_waves(dc_id);
CREATE INDEX idx_pick_waves_date ON pick_waves(wave_date);
CREATE INDEX idx_pick_waves_status ON pick_waves(status);

-- E9b: pick_wave_orders - Orders included in each wave
CREATE TABLE pick_wave_orders (
    wave_id INTEGER NOT NULL REFERENCES pick_waves(id),
    order_id INTEGER NOT NULL REFERENCES orders(id),
    pick_sequence INTEGER,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'picking', 'picked', 'packed', 'staged')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (wave_id, order_id)
);

CREATE INDEX idx_pick_wave_orders_order ON pick_wave_orders(order_id);

-- ============================================================================
-- DOMAIN E2: LOGISTICS (Transport Network)
-- ============================================================================
-- Multi-leg routing: carriers → contracts → rates → segments → routes

-- E2-1: carriers - Carrier profiles with sustainability ratings
CREATE TABLE carriers (
    id SERIAL PRIMARY KEY,
    carrier_code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    carrier_type VARCHAR(30) NOT NULL
        CHECK (carrier_type IN ('trucking', 'rail', 'ocean', 'air', 'parcel', 'ltl', '3pl')),
    scac_code VARCHAR(10),                         -- Standard Carrier Alpha Code
    headquarters_country VARCHAR(100),
    service_regions TEXT[],                        -- Array of regions served
    sustainability_rating VARCHAR(10),             -- A, B, C, D, F
    on_time_delivery_rate DECIMAL(5,4),
    damage_rate DECIMAL(5,4),
    is_preferred BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_carriers_type ON carriers(carrier_type);
CREATE INDEX idx_carriers_preferred ON carriers(is_preferred) WHERE is_preferred = true;
CREATE INDEX idx_carriers_active ON carriers(is_active) WHERE is_active = true;

-- E2-2: carrier_contracts - Rate agreements with date effectivity
CREATE TABLE carrier_contracts (
    id SERIAL PRIMARY KEY,
    contract_number VARCHAR(30) UNIQUE NOT NULL,
    carrier_id INTEGER NOT NULL REFERENCES carriers(id),
    contract_type VARCHAR(30) NOT NULL
        CHECK (contract_type IN ('annual', 'spot', 'dedicated', 'volume_commitment')),
    effective_from DATE NOT NULL,
    effective_to DATE NOT NULL,
    min_volume_commitment DECIMAL(12,2),
    volume_unit VARCHAR(20),                       -- cases, pallets, kg, TEU
    status VARCHAR(20) DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'expired', 'terminated')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_carrier_contracts_carrier ON carrier_contracts(carrier_id);
CREATE INDEX idx_carrier_contracts_dates ON carrier_contracts(effective_from, effective_to);
CREATE INDEX idx_carrier_contracts_status ON carrier_contracts(status);

-- E2-3: carrier_rates - Rate tables by mode, weight break, lane
CREATE TABLE carrier_rates (
    id SERIAL PRIMARY KEY,
    contract_id INTEGER NOT NULL REFERENCES carrier_contracts(id),
    origin_type VARCHAR(20) NOT NULL,              -- plant, dc, port, city, region
    origin_code VARCHAR(50) NOT NULL,
    destination_type VARCHAR(20) NOT NULL,
    destination_code VARCHAR(50) NOT NULL,
    transport_mode VARCHAR(30) NOT NULL
        CHECK (transport_mode IN ('ftl', 'ltl', 'rail', 'ocean_fcl', 'ocean_lcl', 'air', 'parcel', 'intermodal')),
    weight_break_min_kg DECIMAL(10,2) DEFAULT 0,
    weight_break_max_kg DECIMAL(10,2),
    rate_per_kg DECIMAL(10,4),
    rate_per_case DECIMAL(10,4),
    rate_per_pallet DECIMAL(10,4),
    rate_per_shipment DECIMAL(10,2),               -- Flat rate
    fuel_surcharge_percent DECIMAL(5,2) DEFAULT 0,
    currency VARCHAR(3) DEFAULT 'USD',
    transit_days INTEGER,
    effective_from DATE NOT NULL,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_carrier_rates_contract ON carrier_rates(contract_id);
CREATE INDEX idx_carrier_rates_origin ON carrier_rates(origin_type, origin_code);
CREATE INDEX idx_carrier_rates_destination ON carrier_rates(destination_type, destination_code);
CREATE INDEX idx_carrier_rates_mode ON carrier_rates(transport_mode);
CREATE INDEX idx_carrier_rates_effective ON carrier_rates(effective_from, effective_to);

-- E2-4: route_segments - Atomic legs: origin → destination
CREATE TABLE route_segments (
    id SERIAL PRIMARY KEY,
    segment_code VARCHAR(30) UNIQUE NOT NULL,      -- e.g., LANE-SH-LA-001
    origin_type VARCHAR(20) NOT NULL
        CHECK (origin_type IN ('plant', 'dc', 'port')),
    origin_id INTEGER NOT NULL,
    destination_type VARCHAR(20) NOT NULL
        CHECK (destination_type IN ('dc', 'port', 'store')),
    destination_id INTEGER NOT NULL,
    transport_mode VARCHAR(30) NOT NULL
        CHECK (transport_mode IN ('truck', 'rail', 'ocean', 'air', 'intermodal', 'last_mile')),
    distance_km DECIMAL(10,2),
    distance_miles DECIMAL(10,2),
    transit_time_hours DECIMAL(8,2),
    is_seasonal BOOLEAN DEFAULT false,             -- For temporal flickering
    seasonal_months INTEGER[],                     -- Months when active, e.g., {4,5,6,7,8,9}
    capacity_reduction_percent DECIMAL(5,2),       -- Winter storms, etc.
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_route_segments_origin ON route_segments(origin_type, origin_id);
CREATE INDEX idx_route_segments_destination ON route_segments(destination_type, destination_id);
CREATE INDEX idx_route_segments_mode ON route_segments(transport_mode);
CREATE INDEX idx_route_segments_seasonal ON route_segments(is_seasonal) WHERE is_seasonal = true;
CREATE INDEX idx_route_segments_active ON route_segments(is_active) WHERE is_active = true;

-- E2-5: routes - Composed routes (multiple segments)
CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    route_code VARCHAR(30) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    origin_type VARCHAR(20) NOT NULL,
    origin_id INTEGER NOT NULL,
    destination_type VARCHAR(20) NOT NULL,
    destination_id INTEGER NOT NULL,
    total_distance_km DECIMAL(10,2),
    total_transit_hours DECIMAL(8,2),
    total_segments INTEGER,
    is_preferred BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_routes_origin ON routes(origin_type, origin_id);
CREATE INDEX idx_routes_destination ON routes(destination_type, destination_id);
CREATE INDEX idx_routes_preferred ON routes(is_preferred) WHERE is_preferred = true;
CREATE INDEX idx_routes_active ON routes(is_active) WHERE is_active = true;

-- E2-6: route_segment_assignments - Links routes to segments in sequence
CREATE TABLE route_segment_assignments (
    route_id INTEGER NOT NULL REFERENCES routes(id),
    segment_id INTEGER NOT NULL REFERENCES route_segments(id),
    sequence INTEGER NOT NULL,                     -- Order in route
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (route_id, segment_id, sequence)
);

CREATE INDEX idx_route_segment_assignments_segment ON route_segment_assignments(segment_id);

-- E2-7: shipment_legs - Actual execution: which segments used per shipment
CREATE TABLE shipment_legs (
    id SERIAL PRIMARY KEY,
    shipment_id INTEGER NOT NULL REFERENCES shipments(id),
    route_segment_id INTEGER NOT NULL REFERENCES route_segments(id),
    leg_sequence INTEGER NOT NULL,
    transport_mode VARCHAR(30),                    -- Added for vectorized gen
    carrier_id INTEGER REFERENCES carriers(id),
    planned_departure TIMESTAMP,
    planned_arrival TIMESTAMP,
    actual_departure TIMESTAMP,
    actual_arrival TIMESTAMP,
    delay_hours DECIMAL(8,2),
    actual_transit_hours DECIMAL(8,2),
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'departed', 'in_transit', 'arrived', 'exception', 'on_time', 'delayed', 'severely_delayed')),
    freight_cost DECIMAL(10,2),
    tracking_number VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_shipment_legs_shipment ON shipment_legs(shipment_id);
CREATE INDEX idx_shipment_legs_segment ON shipment_legs(route_segment_id);
CREATE INDEX idx_shipment_legs_carrier ON shipment_legs(carrier_id);
CREATE INDEX idx_shipment_legs_status ON shipment_legs(status);

-- ============================================================================
-- DOMAIN E3: ESG/SUSTAINABILITY (2030 KPIs)
-- ============================================================================
-- Carbon tracking: emission factors → shipment emissions → targets

-- E3-1: emission_factors - CO2/km by mode, fuel type, carrier
CREATE TABLE emission_factors (
    id SERIAL PRIMARY KEY,
    transport_mode VARCHAR(30) NOT NULL,
    fuel_type VARCHAR(30) NOT NULL
        CHECK (fuel_type IN ('diesel', 'gasoline', 'electric', 'hybrid', 'lng', 'hfo', 'jet_fuel', 'biofuel')),
    carrier_id INTEGER REFERENCES carriers(id),    -- NULL = default for mode
    co2_kg_per_km DECIMAL(10,6) NOT NULL,
    co2_kg_per_ton_km DECIMAL(10,6),
    source VARCHAR(200),                           -- Data source (EPA, GLEC, etc.)
    effective_from DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_emission_factors_mode ON emission_factors(transport_mode);
CREATE INDEX idx_emission_factors_fuel ON emission_factors(fuel_type);
CREATE INDEX idx_emission_factors_carrier ON emission_factors(carrier_id);
CREATE INDEX idx_emission_factors_effective ON emission_factors(effective_from, effective_to);

-- E3-2: shipment_emissions - Calculated CO2 per shipment
CREATE TABLE shipment_emissions (
    id SERIAL PRIMARY KEY,
    shipment_id INTEGER NOT NULL REFERENCES shipments(id),
    shipment_leg_id INTEGER REFERENCES shipment_legs(id),
    scope VARCHAR(20) NOT NULL
        CHECK (scope IN ('scope_3_cat_4', 'scope_3_cat_9')),  -- Upstream/downstream transport
    co2_kg DECIMAL(12,4) NOT NULL,
    calculation_method VARCHAR(50),                -- distance_based, spend_based, activity_based
    emission_factor_id INTEGER REFERENCES emission_factors(id),
    distance_km DECIMAL(10,2),
    weight_tons DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_shipment_emissions_shipment ON shipment_emissions(shipment_id);
CREATE INDEX idx_shipment_emissions_leg ON shipment_emissions(shipment_leg_id);
CREATE INDEX idx_shipment_emissions_scope ON shipment_emissions(scope);

-- E3-3: supplier_esg_scores - EcoVadis, CDP, ISO14001, SBTi
CREATE TABLE supplier_esg_scores (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
    assessment_date DATE NOT NULL,
    ecovadis_score INTEGER,                        -- 0-100
    ecovadis_medal VARCHAR(20),                    -- platinum, gold, silver, bronze, none
    cdp_climate_score VARCHAR(5),                  -- A, A-, B, B-, C, C-, D, D-, F
    cdp_water_score VARCHAR(5),
    iso_14001_certified BOOLEAN DEFAULT false,
    iso_14001_expiry DATE,
    sbti_committed BOOLEAN DEFAULT false,          -- Science Based Targets initiative
    sbti_validated BOOLEAN DEFAULT false,
    sbti_target_year INTEGER,
    renewable_energy_percent DECIMAL(5,2),
    waste_diversion_percent DECIMAL(5,2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_supplier_esg_supplier ON supplier_esg_scores(supplier_id);
CREATE INDEX idx_supplier_esg_date ON supplier_esg_scores(assessment_date);
CREATE INDEX idx_supplier_esg_sbti ON supplier_esg_scores(sbti_committed) WHERE sbti_committed = true;

-- E3-4: sustainability_targets - Division/account-level carbon targets
CREATE TABLE sustainability_targets (
    id SERIAL PRIMARY KEY,
    target_type VARCHAR(30) NOT NULL
        CHECK (target_type IN ('division', 'account', 'product', 'route', 'supplier')),
    target_entity_id INTEGER NOT NULL,             -- Polymorphic FK
    metric VARCHAR(50) NOT NULL,                   -- co2_per_case, renewable_percent, etc.
    baseline_year INTEGER NOT NULL,
    baseline_value DECIMAL(12,4) NOT NULL,
    target_year INTEGER NOT NULL,
    target_value DECIMAL(12,4) NOT NULL,
    current_value DECIMAL(12,4),
    reduction_percent DECIMAL(5,2),
    status VARCHAR(20) DEFAULT 'on_track'
        CHECK (status IN ('on_track', 'at_risk', 'off_track', 'achieved')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sustainability_targets_type ON sustainability_targets(target_type, target_entity_id);
CREATE INDEX idx_sustainability_targets_metric ON sustainability_targets(metric);
CREATE INDEX idx_sustainability_targets_status ON sustainability_targets(status);

-- E3-5: modal_shift_opportunities - Truck→rail/intermodal opportunities
CREATE TABLE modal_shift_opportunities (
    id SERIAL PRIMARY KEY,
    route_segment_id INTEGER NOT NULL REFERENCES route_segments(id),
    current_mode VARCHAR(30) NOT NULL,
    proposed_mode VARCHAR(30) NOT NULL,
    annual_volume_tons DECIMAL(12,2),
    current_co2_tons DECIMAL(10,2),
    proposed_co2_tons DECIMAL(10,2),
    co2_reduction_tons DECIMAL(10,2),
    cost_impact_usd DECIMAL(12,2),                 -- Positive = savings, negative = cost
    transit_time_impact_hours DECIMAL(8,2),
    feasibility_score DECIMAL(3,2),                -- 0.00 to 1.00
    status VARCHAR(20) DEFAULT 'identified'
        CHECK (status IN ('identified', 'evaluating', 'approved', 'implementing', 'completed', 'rejected')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_modal_shift_segment ON modal_shift_opportunities(route_segment_id);
CREATE INDEX idx_modal_shift_status ON modal_shift_opportunities(status);

-- ============================================================================
-- DOMAIN F: PLAN (Demand & Supply Planning) - SCOR: Plan
-- ============================================================================
-- Planning loop: POS → forecasts → capacity → supply plans → exceptions

-- F1: pos_sales - Point-of-sale signals (actual demand)
-- NOTE: In this simulation, each row represents WEEKLY AGGREGATED sales 
-- per SKU per Store to reflect high-velocity hypermarket throughput (e.g. MegaMart).
CREATE TABLE pos_sales (
    id SERIAL PRIMARY KEY,
    retail_location_id INTEGER NOT NULL REFERENCES retail_locations(id),
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    sale_date DATE NOT NULL,
    sale_week INTEGER GENERATED ALWAYS AS (EXTRACT(WEEK FROM sale_date)::INTEGER) STORED,
    quantity_eaches INTEGER NOT NULL,
    quantity_cases DECIMAL(10,2),
    revenue DECIMAL(12,2),
    currency VARCHAR(3) DEFAULT 'USD',
    is_promotional BOOLEAN DEFAULT false,
    promo_id INTEGER REFERENCES promotions(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pos_sales_location ON pos_sales(retail_location_id);
CREATE INDEX idx_pos_sales_sku ON pos_sales(sku_id);
CREATE INDEX idx_pos_sales_date ON pos_sales(sale_date);
CREATE INDEX idx_pos_sales_week ON pos_sales(sale_week);
CREATE INDEX idx_pos_sales_promo ON pos_sales(is_promotional) WHERE is_promotional = true;

-- F2: demand_forecasts - Statistical/ML forecasts by SKU/location
CREATE TABLE demand_forecasts (
    id SERIAL PRIMARY KEY,
    forecast_version VARCHAR(30) NOT NULL,         -- e.g., "2024-W48-STAT"
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    location_type VARCHAR(20) NOT NULL
        CHECK (location_type IN ('dc', 'account', 'division', 'total')),
    location_id INTEGER,                           -- NULL for total
    forecast_date DATE NOT NULL,
    forecast_week INTEGER,
    forecast_month INTEGER,
    forecast_quantity_cases DECIMAL(12,2) NOT NULL,
    forecast_type VARCHAR(30) NOT NULL
        CHECK (forecast_type IN ('statistical', 'ml_based', 'manual', 'consensus')),
    model_name VARCHAR(100),                       -- e.g., "Prophet", "ARIMA", "XGBoost"
    confidence_lower DECIMAL(12,2),
    confidence_upper DECIMAL(12,2),
    seasonality_factor DECIMAL(5,2) DEFAULT 1.00,
    promo_lift_factor DECIMAL(5,2) DEFAULT 1.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_demand_forecasts_version ON demand_forecasts(forecast_version);
CREATE INDEX idx_demand_forecasts_sku ON demand_forecasts(sku_id);
CREATE INDEX idx_demand_forecasts_location ON demand_forecasts(location_type, location_id);
CREATE INDEX idx_demand_forecasts_date ON demand_forecasts(forecast_date);
CREATE INDEX idx_demand_forecasts_type ON demand_forecasts(forecast_type);

-- F3: forecast_accuracy - MAPE, bias tracking by SKU/account
CREATE TABLE forecast_accuracy (
    id SERIAL PRIMARY KEY,
    forecast_id INTEGER NOT NULL REFERENCES demand_forecasts(id),
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    location_type VARCHAR(20) NOT NULL,
    location_id INTEGER,
    measurement_date DATE NOT NULL,
    forecast_quantity DECIMAL(12,2) NOT NULL,
    actual_quantity DECIMAL(12,2) NOT NULL,
    absolute_error DECIMAL(12,2) GENERATED ALWAYS AS (ABS(forecast_quantity - actual_quantity)) STORED,
    mape DECIMAL(8,4),                             -- Mean Absolute Percentage Error
    bias_percent DECIMAL(8,4),                     -- Positive = over-forecast
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_forecast_accuracy_forecast ON forecast_accuracy(forecast_id);
CREATE INDEX idx_forecast_accuracy_sku ON forecast_accuracy(sku_id);
CREATE INDEX idx_forecast_accuracy_date ON forecast_accuracy(measurement_date);

-- F4: consensus_adjustments - S&OP overrides to statistical forecast
CREATE TABLE consensus_adjustments (
    id SERIAL PRIMARY KEY,
    forecast_id INTEGER NOT NULL REFERENCES demand_forecasts(id),
    adjustment_type VARCHAR(30) NOT NULL
        CHECK (adjustment_type IN ('promo_uplift', 'new_distribution', 'discontinuation', 'event', 'market_intel', 'executive_override')),
    adjustment_percent DECIMAL(8,2),               -- +/- percentage
    adjustment_quantity DECIMAL(12,2),             -- +/- absolute quantity
    reason TEXT NOT NULL,
    approved_by VARCHAR(100),
    approval_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_consensus_adjustments_forecast ON consensus_adjustments(forecast_id);
CREATE INDEX idx_consensus_adjustments_type ON consensus_adjustments(adjustment_type);

-- F5: replenishment_params - Safety stock, reorder points by node
CREATE TABLE replenishment_params (
    id SERIAL PRIMARY KEY,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    location_type VARCHAR(20) NOT NULL
        CHECK (location_type IN ('dc', 'store')),
    location_id INTEGER NOT NULL,
    safety_stock_days DECIMAL(5,2) NOT NULL,
    safety_stock_cases INTEGER,
    reorder_point_cases INTEGER NOT NULL,
    reorder_quantity_cases INTEGER NOT NULL,
    max_stock_cases INTEGER,
    review_cycle_days INTEGER DEFAULT 7,           -- How often to review
    lead_time_days INTEGER NOT NULL,
    service_level_target DECIMAL(5,4) DEFAULT 0.9500,
    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (sku_id, location_type, location_id, effective_from)
);

CREATE INDEX idx_replenishment_params_sku ON replenishment_params(sku_id);
CREATE INDEX idx_replenishment_params_location ON replenishment_params(location_type, location_id);
CREATE INDEX idx_replenishment_params_effective ON replenishment_params(effective_from, effective_to);

-- F6: demand_allocation - How forecasted demand is allocated down the network
CREATE TABLE demand_allocation (
    id SERIAL PRIMARY KEY,
    forecast_id INTEGER NOT NULL REFERENCES demand_forecasts(id),
    from_location_type VARCHAR(20) NOT NULL,       -- Where demand aggregated from
    from_location_id INTEGER,
    to_location_type VARCHAR(20) NOT NULL,         -- Where demand allocated to
    to_location_id INTEGER NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    allocated_quantity_cases DECIMAL(12,2) NOT NULL,
    allocation_percent DECIMAL(5,4),
    allocation_method VARCHAR(30)
        CHECK (allocation_method IN ('historical', 'fair_share', 'priority', 'manual')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_demand_allocation_forecast ON demand_allocation(forecast_id);
CREATE INDEX idx_demand_allocation_from ON demand_allocation(from_location_type, from_location_id);
CREATE INDEX idx_demand_allocation_to ON demand_allocation(to_location_type, to_location_id);
CREATE INDEX idx_demand_allocation_sku ON demand_allocation(sku_id);

-- F7: capacity_plans - Available capacity by plant/line/period
CREATE TABLE capacity_plans (
    id SERIAL PRIMARY KEY,
    plan_version VARCHAR(30) NOT NULL,
    plant_id INTEGER NOT NULL REFERENCES plants(id),
    production_line_id INTEGER REFERENCES production_lines(id),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    available_hours DECIMAL(8,2) NOT NULL,
    planned_hours DECIMAL(8,2) DEFAULT 0,
    utilization_percent DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE WHEN available_hours > 0 THEN (planned_hours / available_hours * 100)::DECIMAL(5,2) ELSE 0 END
    ) STORED,
    available_capacity_cases INTEGER,
    planned_capacity_cases INTEGER DEFAULT 0,
    maintenance_hours DECIMAL(8,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (plan_version, plant_id, production_line_id, period_start)
);

CREATE INDEX idx_capacity_plans_version ON capacity_plans(plan_version);
CREATE INDEX idx_capacity_plans_plant ON capacity_plans(plant_id);
CREATE INDEX idx_capacity_plans_line ON capacity_plans(production_line_id);
CREATE INDEX idx_capacity_plans_period ON capacity_plans(period_start, period_end);

-- F8: supply_plans - Planned production/procurement to meet demand
CREATE TABLE supply_plans (
    id SERIAL PRIMARY KEY,
    plan_version VARCHAR(30) NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    source_type VARCHAR(20) NOT NULL
        CHECK (source_type IN ('production', 'procurement', 'transfer')),
    source_id INTEGER NOT NULL,                    -- plant_id, supplier_id, or dc_id
    destination_type VARCHAR(20) NOT NULL,
    destination_id INTEGER NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    planned_quantity_cases INTEGER NOT NULL,
    committed_quantity_cases INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'planned'
        CHECK (status IN ('planned', 'committed', 'in_progress', 'completed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_supply_plans_version ON supply_plans(plan_version);
CREATE INDEX idx_supply_plans_sku ON supply_plans(sku_id);
CREATE INDEX idx_supply_plans_source ON supply_plans(source_type, source_id);
CREATE INDEX idx_supply_plans_destination ON supply_plans(destination_type, destination_id);
CREATE INDEX idx_supply_plans_period ON supply_plans(period_start, period_end);
CREATE INDEX idx_supply_plans_status ON supply_plans(status);

-- F9: plan_exceptions - Gap identification when demand > capacity
CREATE TABLE plan_exceptions (
    id SERIAL PRIMARY KEY,
    plan_version VARCHAR(30) NOT NULL,
    exception_type VARCHAR(30) NOT NULL
        CHECK (exception_type IN ('capacity_shortage', 'material_shortage', 'lead_time_violation', 'inventory_excess', 'demand_spike', 'supply_disruption')),
    severity VARCHAR(10) NOT NULL
        CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    sku_id INTEGER REFERENCES skus(id),
    location_type VARCHAR(20),
    location_id INTEGER,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    gap_quantity_cases INTEGER,
    gap_percent DECIMAL(5,2),
    root_cause TEXT,
    recommended_action TEXT,
    status VARCHAR(20) DEFAULT 'open'
        CHECK (status IN ('open', 'acknowledged', 'resolving', 'resolved', 'accepted')),
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_plan_exceptions_version ON plan_exceptions(plan_version);
CREATE INDEX idx_plan_exceptions_type ON plan_exceptions(exception_type);
CREATE INDEX idx_plan_exceptions_severity ON plan_exceptions(severity);
CREATE INDEX idx_plan_exceptions_sku ON plan_exceptions(sku_id);
CREATE INDEX idx_plan_exceptions_status ON plan_exceptions(status);

-- ============================================================================
-- DOMAIN G: RETURN (Regenerate) - SCOR: Return
-- ============================================================================
-- Returns flow: RMA → returns → lines → disposition

-- G1: rma_authorizations - Return Merchandise Authorization workflow
CREATE TABLE rma_authorizations (
    id SERIAL PRIMARY KEY,
    rma_number VARCHAR(30) UNIQUE NOT NULL,
    retail_account_id INTEGER NOT NULL REFERENCES retail_accounts(id),
    request_date DATE NOT NULL DEFAULT CURRENT_DATE,
    reason_code VARCHAR(30) NOT NULL
        CHECK (reason_code IN ('damaged', 'expired', 'quality_defect', 'overstock', 'recall', 'wrong_shipment', 'customer_return')),
    status VARCHAR(20) DEFAULT 'requested'
        CHECK (status IN ('requested', 'approved', 'rejected', 'expired')),
    approved_by VARCHAR(100),
    approval_date DATE,
    expiry_date DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rma_account ON rma_authorizations(retail_account_id);
CREATE INDEX idx_rma_status ON rma_authorizations(status);
CREATE INDEX idx_rma_reason ON rma_authorizations(reason_code);
CREATE INDEX idx_rma_date ON rma_authorizations(request_date);

-- G2: returns - Return events
CREATE TABLE returns (
    id SERIAL PRIMARY KEY,
    return_number VARCHAR(30) UNIQUE NOT NULL,
    rma_id INTEGER REFERENCES rma_authorizations(id),
    order_id INTEGER REFERENCES orders(id),
    retail_location_id INTEGER REFERENCES retail_locations(id),
    dc_id INTEGER NOT NULL REFERENCES distribution_centers(id),
    return_date DATE NOT NULL DEFAULT CURRENT_DATE,
    received_date DATE,
    status VARCHAR(20) DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_transit', 'received', 'inspecting', 'processed', 'closed')),
    total_cases INTEGER,
    credit_amount DECIMAL(12,2),
    currency VARCHAR(3) DEFAULT 'USD',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_returns_rma ON returns(rma_id);
CREATE INDEX idx_returns_order ON returns(order_id);
CREATE INDEX idx_returns_location ON returns(retail_location_id);
CREATE INDEX idx_returns_dc ON returns(dc_id);
CREATE INDEX idx_returns_status ON returns(status);
CREATE INDEX idx_returns_date ON returns(return_date);

-- G3: return_lines - Return items with condition assessment
CREATE TABLE return_lines (
    return_id INTEGER NOT NULL REFERENCES returns(id),
    line_number INTEGER NOT NULL,
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    batch_id INTEGER REFERENCES batches(id),
    lot_number VARCHAR(50),
    quantity_cases INTEGER NOT NULL,
    condition VARCHAR(20) NOT NULL
        CHECK (condition IN ('sellable', 'damaged', 'expired', 'contaminated', 'unknown')),
    inspection_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (return_id, line_number)
);

CREATE INDEX idx_return_lines_sku ON return_lines(sku_id);
CREATE INDEX idx_return_lines_batch ON return_lines(batch_id);
CREATE INDEX idx_return_lines_condition ON return_lines(condition);

-- G4: disposition_logs - Restock, scrap, regenerate, donate decisions
CREATE TABLE disposition_logs (
    id SERIAL PRIMARY KEY,
    return_id INTEGER NOT NULL REFERENCES returns(id),
    return_line_number INTEGER NOT NULL,
    disposition VARCHAR(20) NOT NULL
        CHECK (disposition IN ('restock', 'scrap', 'donate', 'rework', 'liquidate', 'quarantine')),
    quantity_cases INTEGER NOT NULL,
    destination_location_type VARCHAR(20),         -- dc, scrap_vendor, charity, etc.
    destination_location_id INTEGER,
    recovery_value DECIMAL(12,2),
    disposal_cost DECIMAL(12,2),
    processed_by VARCHAR(100),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (return_id, return_line_number) REFERENCES return_lines(return_id, line_number)
);

CREATE INDEX idx_disposition_return ON disposition_logs(return_id);
CREATE INDEX idx_disposition_type ON disposition_logs(disposition);

-- ============================================================================
-- DOMAIN H: ORCHESTRATE (Hub) - SCOR: Orchestrate
-- ============================================================================
-- Performance management: KPI thresholds → actuals → OSA → rules → risks

-- H1: kpi_thresholds - Desmet Triangle targets (Service, Cost, Cash)
CREATE TABLE kpi_thresholds (
    id SERIAL PRIMARY KEY,
    kpi_code VARCHAR(30) UNIQUE NOT NULL,
    kpi_name VARCHAR(100) NOT NULL,
    kpi_category VARCHAR(30) NOT NULL
        CHECK (kpi_category IN ('service', 'cost', 'cash', 'quality', 'sustainability')),
    desmet_dimension VARCHAR(20)
        CHECK (desmet_dimension IN ('service', 'cost', 'cash')),
    unit VARCHAR(30) NOT NULL,                     -- percent, days, USD, kg_co2, etc.
    direction VARCHAR(10) NOT NULL
        CHECK (direction IN ('higher', 'lower')),  -- Higher is better or lower is better
    target_value DECIMAL(12,4) NOT NULL,
    warning_threshold DECIMAL(12,4),
    critical_threshold DECIMAL(12,4),
    industry_benchmark DECIMAL(12,4),
    scope VARCHAR(30) DEFAULT 'global'
        CHECK (scope IN ('global', 'division', 'account', 'sku')),
    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kpi_thresholds_category ON kpi_thresholds(kpi_category);
CREATE INDEX idx_kpi_thresholds_desmet ON kpi_thresholds(desmet_dimension);
CREATE INDEX idx_kpi_thresholds_scope ON kpi_thresholds(scope);

-- H2: kpi_actuals - Calculated KPI values vs thresholds
CREATE TABLE kpi_actuals (
    id SERIAL PRIMARY KEY,
    kpi_id INTEGER NOT NULL REFERENCES kpi_thresholds(id),
    measurement_date DATE NOT NULL,
    measurement_week INTEGER,
    measurement_month INTEGER,
    scope_type VARCHAR(30) NOT NULL,               -- global, division, account, sku
    scope_id INTEGER,                              -- NULL for global
    actual_value DECIMAL(12,4) NOT NULL,
    target_value DECIMAL(12,4) NOT NULL,
    variance DECIMAL(12,4) GENERATED ALWAYS AS (actual_value - target_value) STORED,
    variance_percent DECIMAL(8,4),
    status VARCHAR(20)
        CHECK (status IN ('green', 'yellow', 'red')),
    trend VARCHAR(10)
        CHECK (trend IN ('improving', 'stable', 'declining')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_kpi_actuals_kpi ON kpi_actuals(kpi_id);
CREATE INDEX idx_kpi_actuals_date ON kpi_actuals(measurement_date);
CREATE INDEX idx_kpi_actuals_scope ON kpi_actuals(scope_type, scope_id);
CREATE INDEX idx_kpi_actuals_status ON kpi_actuals(status);

-- H3: osa_metrics - On-shelf availability (~500K measurements)
CREATE TABLE osa_metrics (
    id SERIAL PRIMARY KEY,
    retail_location_id INTEGER NOT NULL REFERENCES retail_locations(id),
    sku_id INTEGER NOT NULL REFERENCES skus(id),
    measurement_date DATE NOT NULL,
    measurement_time TIME,
    is_in_stock BOOLEAN NOT NULL,
    shelf_capacity INTEGER,
    shelf_quantity INTEGER,
    days_of_stock DECIMAL(5,2),
    out_of_stock_reason VARCHAR(30)
        CHECK (out_of_stock_reason IN ('dc_stockout', 'delivery_miss', 'shelf_gap', 'planogram_issue', 'demand_spike', 'unknown')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_osa_metrics_location ON osa_metrics(retail_location_id);
CREATE INDEX idx_osa_metrics_sku ON osa_metrics(sku_id);
CREATE INDEX idx_osa_metrics_date ON osa_metrics(measurement_date);
CREATE INDEX idx_osa_metrics_in_stock ON osa_metrics(is_in_stock);
CREATE INDEX idx_osa_metrics_reason ON osa_metrics(out_of_stock_reason) WHERE out_of_stock_reason IS NOT NULL;

-- H4: business_rules - Policy management
CREATE TABLE business_rules (
    id SERIAL PRIMARY KEY,
    rule_code VARCHAR(30) UNIQUE NOT NULL,
    rule_name VARCHAR(200) NOT NULL,
    rule_category VARCHAR(30) NOT NULL
        CHECK (rule_category IN ('allocation', 'pricing', 'fulfillment', 'inventory', 'promotion', 'credit')),
    rule_type VARCHAR(20) NOT NULL
        CHECK (rule_type IN ('constraint', 'priority', 'threshold', 'calculation')),
    scope VARCHAR(30) DEFAULT 'global',
    scope_id INTEGER,
    condition_expression TEXT,                     -- SQL-like condition
    action_expression TEXT,                        -- What happens when triggered
    priority INTEGER DEFAULT 100,                  -- Lower = higher priority
    is_active BOOLEAN DEFAULT true,
    effective_from DATE DEFAULT CURRENT_DATE,
    effective_to DATE,
    created_by VARCHAR(100),
    approved_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_business_rules_category ON business_rules(rule_category);
CREATE INDEX idx_business_rules_type ON business_rules(rule_type);
CREATE INDEX idx_business_rules_active ON business_rules(is_active) WHERE is_active = true;
CREATE INDEX idx_business_rules_priority ON business_rules(priority);

-- H5: risk_events - Risk registry
CREATE TABLE risk_events (
    id SERIAL PRIMARY KEY,
    event_code VARCHAR(30) UNIQUE NOT NULL,
    event_type VARCHAR(30) NOT NULL
        CHECK (event_type IN ('supplier_disruption', 'quality_hold', 'logistics_delay', 'demand_shock', 'capacity_constraint', 'natural_disaster', 'geopolitical', 'recall')),
    severity VARCHAR(10) NOT NULL
        CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    probability DECIMAL(3,2),                      -- 0.00 to 1.00
    impact_score INTEGER,                          -- 1-10
    risk_score DECIMAL(5,2) GENERATED ALWAYS AS (COALESCE(probability, 0.5) * COALESCE(impact_score, 5)) STORED,
    affected_entity_type VARCHAR(30),              -- supplier, plant, dc, route, sku
    affected_entity_id INTEGER,
    description TEXT NOT NULL,
    root_cause TEXT,
    mitigation_plan TEXT,
    status VARCHAR(20) DEFAULT 'identified'
        CHECK (status IN ('identified', 'assessing', 'mitigating', 'monitoring', 'resolved', 'accepted')),
    identified_date DATE DEFAULT CURRENT_DATE,
    target_resolution_date DATE,
    actual_resolution_date DATE,
    owner VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_risk_events_type ON risk_events(event_type);
CREATE INDEX idx_risk_events_severity ON risk_events(severity);
CREATE INDEX idx_risk_events_status ON risk_events(status);
CREATE INDEX idx_risk_events_entity ON risk_events(affected_entity_type, affected_entity_id);
CREATE INDEX idx_risk_events_score ON risk_events(risk_score DESC);

-- H6: audit_logs - Change tracking for governance
CREATE TABLE audit_logs (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id INTEGER NOT NULL,
    action VARCHAR(20) NOT NULL
        CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_fields TEXT[],
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(50),
    user_agent VARCHAR(500)
);

CREATE INDEX idx_audit_logs_table ON audit_logs(table_name);
CREATE INDEX idx_audit_logs_record ON audit_logs(table_name, record_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_time ON audit_logs(changed_at);
CREATE INDEX idx_audit_logs_user ON audit_logs(changed_by);

-- ============================================================================
-- DEFERRED FOREIGN KEYS
-- ============================================================================
-- These FKs reference tables defined later in the schema

-- Domain A references
ALTER TABLE purchase_orders ADD CONSTRAINT fk_po_plant
    FOREIGN KEY (plant_id) REFERENCES plants(id);
ALTER TABLE goods_receipts ADD CONSTRAINT fk_gr_plant
    FOREIGN KEY (plant_id) REFERENCES plants(id);

-- Domain B references
ALTER TABLE plants ADD CONSTRAINT fk_plants_division
    FOREIGN KEY (division_id) REFERENCES divisions(id);
ALTER TABLE formulas ADD CONSTRAINT fk_formulas_product
    FOREIGN KEY (product_id) REFERENCES products(id);

-- Domain D references
ALTER TABLE promotion_accounts ADD CONSTRAINT fk_promo_accounts_account
    FOREIGN KEY (retail_account_id) REFERENCES retail_accounts(id);
ALTER TABLE orders ADD CONSTRAINT fk_orders_account
    FOREIGN KEY (retail_account_id) REFERENCES retail_accounts(id);
ALTER TABLE orders ADD CONSTRAINT fk_orders_location
    FOREIGN KEY (retail_location_id) REFERENCES retail_locations(id);
ALTER TABLE order_allocations ADD CONSTRAINT fk_allocations_dc
    FOREIGN KEY (dc_id) REFERENCES distribution_centers(id);

-- Domain E references
ALTER TABLE shipments ADD CONSTRAINT fk_shipments_carrier
    FOREIGN KEY (carrier_id) REFERENCES carriers(id);
ALTER TABLE shipments ADD CONSTRAINT fk_shipments_route
    FOREIGN KEY (route_id) REFERENCES routes(id);

-- ============================================================================
-- VIEWS: UoM Normalization, Hierarchy Flattening, Shortcut Edges
-- ============================================================================

-- V1: v_location_divisions - Flattened hierarchy for aggregation
CREATE VIEW v_location_divisions AS
SELECT
    rl.id AS location_id,
    rl.location_code,
    rl.name AS location_name,
    ra.id AS account_id,
    ra.account_code,
    ra.name AS account_name,
    d.id AS division_id,
    d.division_code,
    d.name AS division_name
FROM retail_locations rl
JOIN retail_accounts ra ON rl.retail_account_id = ra.id
JOIN divisions d ON ra.division_id = d.id;

-- V2: v_transport_normalized - Normalize distances/costs to base units
CREATE VIEW v_transport_normalized AS
SELECT
    rs.id,
    rs.segment_code,
    rs.origin_type,
    rs.origin_id,
    rs.destination_type,
    rs.destination_id,
    rs.transport_mode,
    -- Normalize to km (if miles provided)
    COALESCE(rs.distance_km, rs.distance_miles * 1.60934) AS distance_km,
    rs.transit_time_hours,
    rs.is_seasonal,
    rs.seasonal_months,
    rs.is_active
FROM route_segments rs;

-- V3: v_batch_destinations - Shortcut edge for recall tracing
CREATE VIEW v_batch_destinations AS
SELECT DISTINCT
    sl.batch_id,
    b.batch_number,
    s.destination_type,
    s.destination_id,
    s.ship_date,
    sl.quantity_cases,
    sl.lot_number
FROM shipment_lines sl
JOIN shipments s ON sl.shipment_id = s.id
JOIN batches b ON sl.batch_id = b.id
WHERE s.status IN ('delivered', 'in_transit');

-- V4: v_inventory_summary - Inventory by location with aging
CREATE VIEW v_inventory_summary AS
SELECT
    i.location_type,
    i.location_id,
    i.sku_id,
    s.sku_code,
    s.name AS sku_name,
    SUM(i.quantity_cases) AS total_cases,
    SUM(i.quantity_reserved) AS reserved_cases,
    SUM(i.quantity_available) AS available_cases,
    MIN(i.expiry_date) AS earliest_expiry,
    MIN(i.days_until_expiry) AS min_days_to_expiry
FROM inventory i
JOIN skus s ON i.sku_id = s.id
GROUP BY i.location_type, i.location_id, i.sku_id, s.sku_code, s.name;

-- V4b: v_inventory_waterfall - Inventory position breakdown by type (safety/cycle/transit)
CREATE VIEW v_inventory_waterfall AS
WITH sku_demand AS (
    SELECT
        ol.sku_id,
        SUM(ol.quantity_cases) / 365.0 AS daily_demand
    FROM order_lines ol
    JOIN orders o ON ol.order_id = o.id
    GROUP BY ol.sku_id
)
SELECT
    COALESCE(dc.name, CASE WHEN i.location_type = 'in_transit' THEN 'In Transit' ELSE 'Unknown' END) AS location_name,
    i.location_type,
    i.sku_id,
    s.sku_code,
    s.name AS sku_name,
    SUM(CASE WHEN i.location_type = 'in_transit' THEN i.quantity_cases ELSE 0 END) AS transit_inventory,
    SUM(CASE WHEN i.inventory_type = 'safety_stock' THEN i.quantity_cases ELSE 0 END) AS safety_stock,
    SUM(CASE WHEN i.inventory_type = 'cycle_stock' THEN i.quantity_cases ELSE 0 END) AS cycle_stock,
    SUM(i.quantity_cases) AS total_position,
    COALESCE(d.daily_demand * 14, 0) AS target_safety_stock,
    ROUND(SUM(i.quantity_cases) / NULLIF(d.daily_demand, 0), 1) AS days_of_supply
FROM inventory i
LEFT JOIN distribution_centers dc ON i.location_type = 'dc' AND i.location_id = dc.id
LEFT JOIN skus s ON i.sku_id = s.id
LEFT JOIN sku_demand d ON i.sku_id = d.sku_id
GROUP BY
    COALESCE(dc.name, CASE WHEN i.location_type = 'in_transit' THEN 'In Transit' ELSE 'Unknown' END),
    i.location_type,
    i.sku_id,
    s.sku_code,
    s.name,
    d.daily_demand;

-- V5: v_supplier_risk - Supplier risk assessment
CREATE VIEW v_supplier_risk AS
SELECT
    s.id AS supplier_id,
    s.supplier_code,
    s.name AS supplier_name,
    s.tier,
    s.country,
    s.qualification_status,
    s.risk_score,
    COUNT(DISTINCT si.ingredient_id) AS ingredient_count,
    COUNT(DISTINCT CASE WHEN si.is_preferred THEN si.ingredient_id END) AS preferred_count,
    AVG(si.on_time_delivery_rate) AS avg_otd_rate,
    AVG(si.quality_acceptance_rate) AS avg_quality_rate,
    MAX(esg.ecovadis_score) AS latest_ecovadis_score,
    MAX(esg.sbti_committed::int) AS sbti_committed
FROM suppliers s
LEFT JOIN supplier_ingredients si ON s.id = si.supplier_id
LEFT JOIN supplier_esg_scores esg ON s.id = esg.supplier_id
WHERE s.is_active = true
GROUP BY s.id, s.supplier_code, s.name, s.tier, s.country, s.qualification_status, s.risk_score;

-- V6: v_single_source_ingredients - SPOF detection (ingredients with only 1 supplier)
CREATE VIEW v_single_source_ingredients AS
SELECT
    i.id AS ingredient_id,
    i.ingredient_code,
    i.name AS ingredient_name,
    i.category,
    COUNT(si.supplier_id) AS supplier_count,
    MIN(s.supplier_code) AS sole_supplier_code,
    MIN(s.name) AS sole_supplier_name,
    MIN(s.country) AS sole_supplier_country
FROM ingredients i
LEFT JOIN supplier_ingredients si ON i.id = si.ingredient_id AND si.is_approved = true
LEFT JOIN suppliers s ON si.supplier_id = s.id AND s.is_active = true
WHERE i.is_active = true
GROUP BY i.id, i.ingredient_code, i.name, i.category
HAVING COUNT(si.supplier_id) = 1;

-- V7: v_order_fulfillment_metrics - OTIF calculation
CREATE VIEW v_order_fulfillment_metrics AS
SELECT
    o.id AS order_id,
    o.order_number,
    o.order_date,
    o.requested_delivery_date,
    o.actual_delivery_date,
    o.status,
    CASE
        WHEN o.actual_delivery_date <= o.requested_delivery_date THEN true
        ELSE false
    END AS on_time,
    CASE
        WHEN o.status = 'delivered' AND NOT EXISTS (
            SELECT 1 FROM order_lines ol
            WHERE ol.order_id = o.id AND ol.status != 'shipped'
        ) THEN true
        ELSE false
    END AS in_full,
    CASE
        WHEN o.actual_delivery_date <= o.requested_delivery_date
        AND o.status = 'delivered'
        AND NOT EXISTS (
            SELECT 1 FROM order_lines ol
            WHERE ol.order_id = o.id AND ol.status != 'shipped'
        ) THEN true
        ELSE false
    END AS perfect_order
FROM orders o
WHERE o.status IN ('delivered', 'shipped');

-- V8: v_dc_utilization - DC capacity utilization
CREATE VIEW v_dc_utilization AS
SELECT
    dc.id AS dc_id,
    dc.dc_code,
    dc.name AS dc_name,
    dc.division_id,
    dc.capacity_cases,
    COALESCE(inv.current_cases, 0) AS current_cases,
    CASE
        WHEN dc.capacity_cases > 0
        THEN (COALESCE(inv.current_cases, 0)::decimal / dc.capacity_cases * 100)::decimal(5,2)
        ELSE 0
    END AS utilization_percent
FROM distribution_centers dc
LEFT JOIN (
    SELECT location_id, SUM(quantity_cases) AS current_cases
    FROM inventory
    WHERE location_type = 'dc'
    GROUP BY location_id
) inv ON dc.id = inv.location_id
WHERE dc.is_active = true;

-- ============================================================================
-- SUMMARY: 67 Tables + 8 Views
-- ============================================================================
--
-- DOMAIN A - SOURCE (8 tables):
--   ingredients, suppliers, supplier_ingredients, certifications,
--   purchase_orders, purchase_order_lines, goods_receipts, goods_receipt_lines
--
-- DOMAIN B - TRANSFORM (9 tables):
--   plants, production_lines, formulas, formula_ingredients,
--   work_orders, work_order_materials, batches, batch_ingredients, batch_cost_ledger
--
-- DOMAIN C - PRODUCT (5 tables):
--   products, packaging_types, skus, sku_costs, sku_substitutes
--
-- DOMAIN D - ORDER (7 tables):
--   channels, promotions, promotion_skus, promotion_accounts,
--   orders, order_lines, order_allocations
--
-- DOMAIN E - FULFILL (11 tables):
--   divisions, distribution_centers, ports, retail_accounts, retail_locations,
--   shipments, shipment_lines, inventory, pick_waves, pick_wave_orders
--
-- DOMAIN E2 - LOGISTICS (7 tables):
--   carriers, carrier_contracts, carrier_rates, route_segments,
--   routes, route_segment_assignments, shipment_legs
--
-- DOMAIN E3 - ESG/SUSTAINABILITY (5 tables):
--   emission_factors, shipment_emissions, supplier_esg_scores,
--   sustainability_targets, modal_shift_opportunities
--
-- DOMAIN F - PLAN (9 tables):
--   pos_sales, demand_forecasts, forecast_accuracy, consensus_adjustments,
--   replenishment_params, demand_allocation, capacity_plans, supply_plans, plan_exceptions
--
-- DOMAIN G - RETURN (4 tables):
--   rma_authorizations, returns, return_lines, disposition_logs
--
-- DOMAIN H - ORCHESTRATE (6 tables):
--   kpi_thresholds, kpi_actuals, osa_metrics, business_rules, risk_events, audit_logs
--
-- VIEWS (8):
--   v_location_divisions, v_transport_normalized, v_batch_destinations,
--   v_inventory_summary, v_supplier_risk, v_single_source_ingredients,
--   v_order_fulfillment_metrics, v_dc_utilization
--
-- ============================================================================
