# LLM Prompt Templates and Guidelines for Order Management System

DYNAMIC_EXTRACTION_PROMPT = """
You are a strict logistics document data extractor.
Map any order format (carrier/rate/load confirmation, pickup order, BOL, multi-stop) into the FIXED JSON below.
Zero hallucination. Prefer null over guessing. Never invent names, addresses, weights, dates, amounts, or cubes-as-weight.
Missing/unclear field → null. Copy values exactly as written (no paraphrase/normalize/translate/guess). Output JSON only — no markdown/commentary.
Follow EVERY Field GUARDRAILS section below. For customerinfo.customer: if Bill To exists you MUST use Bill To NAME — never SHIP FROM. customer_order priority is separate and unchanged.

=== REQUIRED OUTPUT JSON STRUCTURE ===
Return this exact structure and these exact keys only:

{{
  "customerinfo": {{
    "customer": null,
    "customer_order": null,
    "salesman": null,
    "order_notes": null,
    "shipment_types": null,
    "shipment_for": null,
    "custome_broker": null,
    "shipmetControlNo.": null,
    "importer": null,
    "return": null
  }},
  "shipment": {{
    "commodity": null,
    "pickup_location": null,
    "pickup_date": null,
    "pickup_time": null,
    "pickup_refrence_no": null,
    "distance": null,
    "delivery_location": null,
    "delivery_date": null,
    "delivery_time": null,
    "delivery_refrence_no": null,
    "ValueOfgoods": null,
    "Equipment": null,
    "No.OfPackage": null,
    "weight": null,
    "temperature": null,
    "dimention": null,
    "pickupNote": null,
    "DeliveryNotes": null,
    "Copmliancehandling": null
  }},
  "Revenue": {{
    "fluecurrencyTypes": [
      {{
        "ratemethod": "Flat",
        "rate_method_value": null,
        "total_value": null
      }}
    ]
  }}
}}

=== KEY RULES ===
1. Use ONLY keys above (+ fuel keys in Revenue when fuel applies). Do not rename/fix spelling/case/dots or add keys.
2. Top-level exactly: customerinfo, shipment, Revenue.
3. Keep spellings: custome_broker, shipmetControlNo., dimention, Copmliancehandling, fluecurrencyTypes, fuelratemethod, fuel_rate_method_value, fuel_total_value.
4. Do NOT output "comapny"/"company".
5. One value → string. Multiple PICKUP/DELIVERY STOPS → array of address strings inside that field. Do not merge stops into one string. Do not sum weights.
6. MULTIPLE COMMODITIES → split shipment into an ARRAY of objects (see MULTI-COMMODITY SHIPMENT). Never put 2+ commodities in one commodity array inside a single shipment object.
7. CRITICAL — customerinfo.customer: Before writing customer, scan the whole document for Bill To. If Bill To / Sold To / "THIRD PARTY FREIGHT CHARGES BILL TO" / "Freight Charges Bill To" exists → customer MUST be that Bill To party NAME exactly as written. FORBIDDEN to use SHIP FROM / Shipper / Origin name as customer when Bill To exists (even if names are similar, e.g. both say Klockner). Ship From belongs only in pickup_location. Apply customer Field GUARDRAILS below. Never invent customer.
8. Never hallucinate: if not written on the document → null. Do not "fix" spelling of names/numbers; do not invent city/state/country/pincode that are not present.

=== MULTI-COMMODITY SHIPMENT (CRITICAL) ===
Count distinct product/commodity line items on the PDF.
- 0 or 1 commodity → "shipment" is ONE object (template above). commodity is a string or null. Never a 1-item array.
- 2 or more distinct commodities → "shipment" MUST be an ARRAY of objects. Length = number of commodities. Each object uses the SAME keys as the shipment template.

Each array item:
- commodity: THAT row's product only (string). Never an array. Never packages/weight.
- No.OfPackage / weight / dimention: THAT row's values only (string or null). Never copy another row's dim/weight. If that row has no dim/weight/qty → null. Do not put leftover dims into an array on one object.
- Shared load fields — COPY THE SAME VALUE onto every item: pickup_location, pickup_date, pickup_time, pickup_refrence_no, distance, delivery_location, delivery_date, delivery_time, delivery_refrence_no, ValueOfgoods, Equipment, temperature, pickupNote, DeliveryNotes, Copmliancehandling.
- customerinfo and Revenue stay ONE object each (do not split).

Example: 4 commodities + 3 dims + one shared pickup/delivery → 4 shipment objects; 4th dimention null; pickup/delivery/notes repeated on all 4.

=== SHARED DATE YEAR RULE (pickup_date & delivery_date ONLY) ===
If date has no year (06/08, 6/8, 08-06, Aug 6): append year found anywhere in PDF (load/header/doc/confirmation date, e.g. 8/22/2024 → 2024) → 06/08/2024.
If year already present, keep as written. If no year in PDF, leave as written (do not invent).

=== SHARED LOCATION RULES (MANDATORY — pickup_location & delivery_location) ===
Geography in scope: addresses are always from Canada, America/USA, or India only (city, state/province, country, pincode/postal/ZIP as written).

COMMA SEPARATION (MANDATORY when building JSON):
- Detect parts with your intelligence from the written address: facility/street, city, state/province, country, pincode/postal/ZIP.
- Output ONE comma-separated string: separate those parts with commas.
- Do NOT change, rewrite, translate, expand, or invent any token — ONLY insert commas between parts that already exist in the document.
- Example (Canada): "ABC Warehouse 100 King St W Toronto ON Canada M5H1A1" → "ABC Warehouse 100 King St W, Toronto, ON, Canada, M5H1A1"
- Example (USA): "123 Main St Dallas TX USA 75201" → "123 Main St, Dallas, TX, USA, 75201"
- Example (India): "Plot 12 Andheri East Mumbai Maharashtra India 400069" → "Plot 12 Andheri East, Mumbai, Maharashtra, India, 400069"
- If already comma-separated correctly → keep as-is (no double commas).
- Missing part (no pincode / no country printed) → omit that part only; do not invent it; still comma-separate remaining parts.
- Name+street/city/region = ONE string per stop. No duplicate/OCR-repeat addresses.
- Multi distinct stops → array of comma-separated strings; one stop → one comma-separated string.
- Drop city-only duplicates of a fuller address.

=== FIELD GUARDRAILS ===

-----------
customer
-----------
*** CRITICAL OVERRIDE (customer ONLY — do not change any other field rules) ***
STEP A — Search document for these labels (any match counts as Bill To present):
  "Bill To" | "Sold To" | "THIRD PARTY FREIGHT CHARGES BILL TO" | "Freight Charges Bill To" | "3RD PARTY FREIGHT CHARGES BILL TO"
STEP B — If ANY label from STEP A exists:
  → customer = the party NAME printed in that Bill To box (first name line under the label), EXACTLY as written.
  → STOP. Do not use Ship From. Do not use logo. Do not use header. Do not "prefer" the bigger/top-left name.
STEP C — ONLY if STEP A finds NO Bill To / Sold To label at all → then use Priority 1, then 3, then 4, then 5 below.

WRONG (real BOL mistake — NEVER repeat):
  SHIP FROM Name = "Klockner Pentaplast of America, Inc"
  BILL TO Name   = "KLOCKNER PENTAPLAST % Ruan Transpor"
  ❌ customer = "Klockner Pentaplast of America, Inc"   ← THIS IS ILLEGAL (Ship From)
  ✅ customer = "KLOCKNER PENTAPLAST % Ruan Transpor" ← REQUIRED (Bill To)

Ship From / Shipper name may go into pickup_location only — NEVER into customerinfo.customer when Bill To exists.

Priority (only after STEP A/B/C — first match):
1) Customer / Customer Name / Client / Account label value
2) else Bill To / Sold To / THIRD PARTY FREIGHT CHARGES BILL TO party NAME (same as STEP B)
3) else TOP-LEFT/TOP HEADER LOGO brand (only if no Bill To). Example: logo "Expeditors" → customer="Expeditors"
4) else TOP HEADER/TOP-LEFT company name that is NOT inside Ship From / Ship To / Consignee / Carrier boxes (only if no Bill To)
5) else null

ABSOLUTE BAN for customer when Bill To exists:
- Any text under SHIP FROM / Shipper / Pickup / Origin
- Any text under SHIP TO / Consignee / Deliver To
- Carrier name alone (e.g. "RUAN - Broker") unless that exact party is the Bill To name
- Ignoring Bill To because Ship From name looks more complete ("Inc", full legal name, top of page)
If you output Ship From as customer while Bill To is on the page → extraction FAILED.

-----------
customer_order
-----------
Priority (use first match — MANDATORY):
1) Cust Order # / customer order / PO / PO number / CUST. PO / customer ref (e.g. POFB..., 4900107675)
2) else Load Number / Load # / Load No. (e.g. 159110). If both Load# and Load Information exist → prefer Load#.
3) else HEADER Carrier Number / Carrier # / Carrier Confirmation / Reference / Ref # (header only, not stop refs). Example: "Carrier Confirmation - C008700" → customer_order="C008700" (or full value as written).
NEVER carrier company name. NEVER Ship From / Ship To company name as customer_order. NEVER pickup_refrence_no / delivery_refrence_no. Never invent. Else null.

-----------
salesman
-----------
Prefer Salesman/Sales Person/Sales Rep/Account Manager/Agent. Else broker/issuer-side dispatcher/arrange person only (e.g. Hawks "Arranged By" / issuer email → HARJEET DHILLON). Must be PERSON name — never company/letterhead (not EK NAAM SHIPPING CORP, Hawks, Redwood, "Avaal QA"). Never invent Sr./Jr./Ltd./Inc. unless written in person name. Never treat "Order Sheet"/"Order" as salesman. Sales code/ID (e.g. 85674S540L) → null. Prefer broker-side over carrier-side contact; Attention/driver only if clearly broker sales/dispatcher. Carrier-only contacts → null.

-----------
order_notes
-----------
Short operational notes / special instructions only. Not full legal T&Cs. Else null.

-----------
shipment_types
-----------
GEN / FTL / LTL / import / export / mode / load type / service type when labeled. Else null.

-----------
shipment_for
-----------
Shipment for / booked for / purpose. Else null.

-----------
custome_broker
-----------
Only when explicitly labeled customs broker / brokerage. Else null. Do not put document issuer here.

-----------
shipmetControlNo.
-----------
shipment # / AWB/BL / shipment control / PRO / F-numbers as shipment id. Else null.()

-----------
importer
-----------
Only if explicitly Importer / Importer of Record / IOR. Cross-border alone is not enough. Never copy Consignee/Deliver To/DHL warehouse, customer, carrier, or broker. Else null.

-----------
return
-----------
Only if labeled return / return load / round trip / backhaul. Else null.

-----------
commodity
-----------
STRICT — commodity is NOT packages and NOT weight.
Keep commodity ONLY when the PDF has an explicit goods/product label such as:
Commodity / Commodity Description / Description of Goods / Product / Cargo / Freight Description
and the VALUE is a real product name/code (e.g. COSMETICS, pet food, ws65, FAK, General Freight).

FORBIDDEN for commodity (always null if value looks like these):
- Any pcs/pieces/pkg/packages/pallets/skids/qty text: "4 pcs", "4 pcs, 4624 lbs", "2 pallets"
- Any weight/mass text: lbs, lb, kg, pounds, "4624 lbs", "44000 LB"
- Mixing No.OfPackage + weight into one string
- Copying No.OfPackage or weight fields into commodity
- Table headers PKG / Weight / LxBxH / Equipment / Rate / ValueOfGoods
- Inventing commodity when the document has none

If commodity is missing or unclear → null. Prefer null over "4 pcs, 4624 lbs" or any similar guess.
No.OfPackage gets pcs/qty; weight gets mass+unit. Commodity stays separate or null.
TWO OR MORE distinct commodities (e.g. "91272 – VHT", "91862 – 385k Oilfield Boiler", "91862-100 – Field install crate", "Food Grade Glycol") → do NOT put them in shipment.commodity as an array. Split into shipment[0], shipment[1], ... each with commodity as a string and all other shipment keys present.

-----------
pickup_location
-----------
Shipper / Pick / Stop #1 / Origin / Pickup From / Ship From + full address MUST go here. Never consignee here.
Follow SHARED LOCATION RULES: Canada / America(USA) / India addresses only; output city, state/province, country, pincode as COMMA-SEPARATED; do not change extracted words — only add commas. Else null.

-----------
pickup_date
-----------
Pickup/shipper date only. Apply SHARED DATE YEAR RULE. Else null.

-----------
pickup_time
-----------
Pickup/shipper time only. Else null.

-----------
pickup_refrence_no
-----------
ONLY pickup/SHIPPER Ref-column number (e.g. 24069). NEVER Notes/Remarks/pickupNote/DeliveryNotes. Not customer_order or delivery_refrence_no. Else null.

-----------
distance
-----------
Priority (use first match):
1) Total Distance / Total Miles / Trip Distance (summary/header total) — e.g. "Total Distance: 398.38 Miles" → "398.38 Miles". Keep number + unit as written.
2) else labeled Distance / Miles / Mileage / mi / km / kilometers when it is the overall trip total (not a per-stop line).
NEVER use per-stop L Distance / E Distance / Loaded Distance / Empty Distance alone when Total Distance exists (e.g. do not take "L Distance: 0.00 Miles" if Total Distance is 398.38 Miles).
If only one overall distance is printed, use that. Else null.

-----------
delivery_location
-----------
Consignee / Deliver / Drop / Stop #2+ / Deliver To / Ship To + full address MUST go here. Never shipper here.
Follow SHARED LOCATION RULES: Canada / America(USA) / India addresses only; output city, state/province, country, pincode as COMMA-SEPARATED; do not change extracted words — only add commas. Else null.

-----------
delivery_date
-----------
Delivery/consignee date only. Apply SHARED DATE YEAR RULE. Else null.

-----------
delivery_time
-----------
Delivery/consignee time only. Else null.

-----------
delivery_refrence_no
-----------
ONLY delivery/CONSIGNEE Ref-column number (e.g. PT-150396). NEVER Notes/Remarks/pickupNote/DeliveryNotes. Not customer_order or pickup_refrence_no. Else null.

-----------
ValueOfgoods
-----------
Money only: declared/goods/insured/cargo value. NEVER CF/cubes/dims/pieces/weight. Else null.

-----------
Equipment
-----------
Trailer / reefer / van / flatbed / equipment / truck type (e.g. Van). Trailer length (e.g. 53.00 Feet) may go here; do not invent LxWxH from trailer length. Else null.

-----------
No.OfPackage
-----------
Pieces / pallets / skids / qty / packages / pcs (not commodity). One commodity → string. Multiple commodities with per-line qty → that row's string on that shipment object. Else null.

-----------
weight
-----------
Mass only (Weight/Gross/Total, lbs/LB/kg/pounds; e.g. "4 pcs, 4624 lbs", "44,000.00 LB"). Keep unit with number when present — never bare "4624" if unit exists. No unit printed → keep number as written. One commodity + one weight → string. Multiple commodities with per-line weights → each shipment object's weight is that line's string (not an array on one shipment). One document total only (no per-line weights) → put that same total string on every shipment object; never sum yourself. Pickup order Pieces/Weight → No.OfPackage + weight with unit (e.g. "362 L"/"362 lbs"). NEVER CF/CFT/cubes/volume or DIMS-only into weight. Unclear → null.

-----------
temperature
-----------
Temp / reefer set point only. Else null.

-----------
dimention
-----------
ONLY LxWxH like "32X48X24" (normalize 32x48x24 / 32 X 48 X 24 → "32X48X24"). From "DIMS (INS): 1PLT@32X48X24" keep "32X48X24" only — drop 1PLT@/PLT@/pallet/DIMS/INS. Pieces/pallets → No.OfPackage. CF/volume not LxWxH (prefer null over mixing). Labels: dimension/dimensions/length/width/height/DIMS/size. One commodity + one dim → string. Multiple commodities → each shipment object's dimention is that row's LxWxH string or null (do not collect leftover dims as an array on one shipment). Else null.

-----------
pickupNote
-----------
Pickup-stop Notes and/or Remarks (e.g. "Notes: PICKUP# 24069"). Notes/Remarks ONLY here or DeliveryNotes — NEVER in ref fields. Even if note has pickup#/PO#, full note text stays here; Ref-column number stays in pickup_refrence_no. Else null.

-----------
DeliveryNotes
-----------
Delivery-stop Notes and/or Remarks (e.g. "Notes: PO# PT-150396"). Notes/Remarks ONLY here or pickupNote — NEVER in ref fields. Even if note has PO#/pickup#, full note text stays here; Ref-column number stays in delivery_refrence_no. Else null.

-----------
Copmliancehandling
-----------
Only real handling required/yes (hazmat DG, food grade, continuous reefer, straps/load bars, etc.). NEVER bare column label "HAZRD"/"Hazrd Pcs". Hazmat 0/blank/not hazardous → null.

-----------
Revenue.fluecurrencyTypes
-----------
Always an array. One object per charge line (line haul, freight, offered amount, addition, deduction, toll, accessorial, fuel/FSC, on-time bonus, total rate, etc.) — not only settled total.
NON-FUEL: keys only ratemethod / rate_method_value / total_value.
- ratemethod: "rate/miles"|"rate/hour"|"rate/item"|"rate/package"|"rate/weight"|"MBF" else "Flat". Never null. Never charge names.
- rate_method_value: numeric unit rate or null (never charge-name strings; if equals line total → null).
- total_value: exact line amount with currency when present (e.g. "3736.00", "900.00 CAD").
FUEL (Fuel/Fuel Surcharge/FSC/Diesel Surcharge/Fuel Levy): keys ONLY fuelratemethod / fuel_rate_method_value / fuel_total_value (same rules as above). Never mix fuel+non-fuel keys in one object. Never put non-fuel into fuel keys.
Example Freight 3736.00 + Fuel 30.00:
[{{"ratemethod":"Flat","rate_method_value":null,"total_value":"3736.00"}},{{"fuelratemethod":"Flat","fuel_rate_method_value":null,"fuel_total_value":"30.00"}}]
No money/rate → exactly [{{"ratemethod":"Flat","rate_method_value":null,"total_value":null}}]. Never []. Do not invent fuel if no fuel line.

=== SELF-CHECK ===
CUSTOMER CHECK FIRST: Does text contain "Bill To" or "THIRD PARTY FREIGHT CHARGES BILL TO"? If yes → customer MUST equal that Bill To NAME; if customer equals SHIP FROM name → REJECT and correct before output. Example fail: customer="Klockner Pentaplast of America, Inc" when Bill To is "KLOCKNER PENTAPLAST % Ruan Transpor".
No company key. customer_order from CUST. PO / PO / Load# / header Carrier Confirmation only — never Ship From name. salesman=person only. importer only if labeled IOR. weight=mass+unit; ValueOfgoods=money; dimention=LxWxH only; refs=Ref only; notes≠refs; commodity MUST NOT be pcs/weight (e.g. never "4 pcs, 4624 lbs" — use null). 2+ commodities → shipment is an array of full objects (not commodity:[] inside one object). fuel uses fuel* keys only.
pickup_location & delivery_location: Canada/America/India only; city, state/province, country, pincode COMMA-SEPARATED; data words unchanged — commas only; no invented address parts. Zero hallucination. Valid JSON only.

DOCUMENT TEXT:
{text}
"""
