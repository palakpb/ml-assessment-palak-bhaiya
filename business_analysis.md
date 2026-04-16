# Business Analysis: Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable**

The target variable is `items_sold` — the total number of items sold at a given store in a given month under a given promotion type.

**Candidate Input Features**

| Feature Category | Examples |
|---|---|
| Store attributes | Store ID, location type (urban/semi-urban/rural), store size (sq ft), monthly footfall, local competition density |
| Customer demographics | Average age band, income tier, gender split, loyalty membership rate |
| Promotion type | One-hot encoded: Flat Discount, BOGO, Free Gift, Category-Specific Offer, Loyalty Points Bonus |
| Calendar/temporal | Month, quarter, year, is_weekend flag, is_festival flag, season |
| Historical performance | Rolling 3-month average items sold, prior-month items sold, historical response rate per promotion type per store |
| Interaction features | Store type × promotion type, month × promotion type |

**Type of ML Problem and Justification**

This is a **supervised regression** problem. Given a store, a month, and a candidate promotion, the model predicts a continuous numeric output — expected items sold. This framing naturally supports a **recommendation system**: for each store each month, all five promotion types are scored and the one with the highest predicted volume is recommended.

An alternative framing as multi-class classification (predict *which* promotion wins) is less suitable because it loses the magnitude of difference between options and cannot communicate *how much better* one promotion is than another — information the marketing team needs for prioritisation and budget decisions.

---

### B1(b) — Why Items Sold is a Better Target Variable Than Revenue

**The case for sales volume**

Revenue is the product of price × quantity. Using revenue as the target conflates the effect of the promotion with the effect of price changes, product mix, and discount depth, all of which vary independently of how well the promotion drove customer behaviour. For example, a Flat Discount reduces unit prices, so a promotion that doubles footfall and volume could still show *lower* revenue than a Loyalty Points Bonus that sold fewer but higher-priced items. A model trained on revenue would systematically undervalue discount-based promotions relative to their true behavioural impact.

Items sold isolates the demand-generation effect of a promotion from pricing mechanics. It is also more stable across time: revenue is sensitive to markdown cycles, supplier price changes, and inflation, which introduce non-stationarity that degrades model quality.

**Broader principle: target variable selection**

This illustrates the principle of **aligning the target variable with the causal mechanism you wish to model**, not with the business's ultimate financial metric. The financial outcome (revenue, profit) often depends on many factors the model does not control. Choosing a target closer to the causal chain — the customer's response to a promotion stimulus — produces a more reliable, interpretable model. The business metric can then be recovered downstream by multiplying predicted volume by known prices, keeping the model itself clean.

---

### B1(c) — Alternative Modelling Strategy: Hierarchical / Segmented Models

A single global model is problematic because a promotion that works well in a large urban store with high footfall and high competition may be entirely ineffective in a small rural store with loyal repeat customers. Pooling all stores masks these structural differences and produces recommendations regressed to the mean.

**Proposed strategy: Clustered store segmentation with segment-level models**

1. **Cluster stores** into 3–5 segments based on structural attributes (location type, store size, footfall tier, competition density) using K-means or hierarchical clustering. This is done once as a preprocessing step, not retrained monthly.

2. **Train one model per segment.** Each segment model learns the promotion-response patterns relevant to stores with similar operating contexts. This reduces variance within each model's training data and produces sharper, more transferable recommendations.

3. **Optional refinement — store-level fine-tuning:** For stores with sufficient historical data (e.g., 18+ months), a store-specific residual correction layer (e.g., a Ridge regression on store-fixed-effect residuals) can be added on top of the segment model to capture idiosyncratic patterns.

This approach balances **data efficiency** (segment models have more training rows than individual store models) with **specificity** (each model is not distorted by dissimilar stores). It is also more robust to new store onboarding: a new store is assigned to a segment immediately and receives sensible recommendations before accumulating its own history.

---

## B2. Data and EDA Strategy

### B2(a) — Table Joins, Grain, and Aggregations

**Table overview**

| Table | Key fields |
|---|---|
| `transactions` | `transaction_id`, `store_id`, `date`, `item_id`, `quantity`, `revenue` |
| `store_attributes` | `store_id`, `location_type`, `size_sqft`, `footfall_monthly`, `competition_density`, demographic fields |
| `promotion_details` | `store_id`, `month`, `year`, `promotion_type` |
| `calendar` | `date`, `is_weekend`, `is_festival`, `month`, `year`, `season` |

**Join sequence**

```
transactions
  LEFT JOIN calendar          ON transactions.date = calendar.date
  LEFT JOIN store_attributes  ON transactions.store_id = store_attributes.store_id
  LEFT JOIN promotion_details ON transactions.store_id = promotion_details.store_id
                              AND MONTH(transactions.date) = promotion_details.month
                              AND YEAR(transactions.date)  = promotion_details.year
```

The join to `promotion_details` uses a month-year match because one promotion runs per store per month; the join to `calendar` is at the daily level to preserve the festival and weekend flags before aggregation.

**Grain of the modelling dataset**

One row = one store × one calendar month × one promotion type.

This is the decision unit: the model is asked "given this store, this month, and this promotion, how many items will be sold?"

**Aggregations before modelling**

From the transaction-level joined table, aggregate to the store-month grain:

- `items_sold` = SUM(quantity) → **target variable**
- `total_revenue` = SUM(revenue)
- `transaction_count` = COUNT(DISTINCT transaction_id)
- `avg_basket_size` = items_sold / transaction_count
- `festival_days_in_month` = COUNT(DISTINCT date WHERE is_festival = 1)
- `weekend_days_in_month` = COUNT(DISTINCT date WHERE is_weekend = 1)
- Rolling features (computed after aggregation): 3-month rolling mean of items_sold per store, lagged items_sold (t-1, t-2)

Store attributes are static and joined directly without aggregation. Promotion type is already at month grain.

---

### B2(b) — EDA Strategy

**Analysis 1: Items sold distribution by promotion type (box plots)**

Plot the distribution of monthly items sold for each of the five promotion types, overall and broken out by location type. Look for: which promotion has the highest median, which has the widest variance, and whether BOGO or Free Gift creates strong outliers. Findings will indicate whether promotion type is a strong signal and whether interaction features with location type are warranted.

**Analysis 2: Time series of items sold per store (line plots)**

Plot monthly items sold for a sample of stores across the three years. Look for: trend (long-term growth or decline), seasonality (peaks in festival months, troughs in off-season), and structural breaks (store renovation, competitor entry). Findings will determine whether time features (month-of-year, quarter) and trend variables need to be included, and whether data needs detrending.

**Analysis 3: Promotion assignment heatmap (store × month)**

Create a heatmap showing which promotion was run in each store in each month. Look for: whether certain promotions are always deployed in certain stores (confounding), whether any promotion is rarely used (limited training data), and whether promotions rotate or cluster. Findings affect how confidently the model can learn counterfactual effects — if Store 5 has only ever run Flat Discount, the model cannot reliably score other promotions for it.

**Analysis 4: Correlation matrix and scatter plots of continuous features vs items sold**

Compute Pearson correlation between items sold and candidate numeric features (footfall, store size, competition density, festival days, prior-month items). Plot scatter plots for the top correlated features. Look for: linearity vs non-linearity, outliers driven by data errors (e.g., footfall = 0), and multicollinearity between features (e.g., footfall and store size). Findings guide feature selection, transformation (log-scaling footfall), and whether tree-based vs linear models are more appropriate.

---

### B2(c) — Handling the 80% No-Promotion Imbalance

**How this affects the model**

The model is trained to predict items sold. If 80% of rows have no promotion, the model's gradient updates are dominated by the non-promotion distribution. The model will learn the baseline demand signal well but will underfit the promotion-specific effects — particularly for rare promotion types. The predicted uplift from promotions will be systematically underestimated, and differences between promotion types will be compressed toward zero, degrading ranking quality.

**Steps to address it**

1. **Separate no-promotion baseline model from promotion-uplift model.** Train one model to predict baseline items sold (no-promotion months) and a second model to predict the uplift (items sold during promotion minus predicted baseline). The recommendation engine scores uplift across promotion types. This cleanly separates the two estimation problems.

2. **Oversample promotion rows** using SMOTE or simple random oversampling before training a unified model, so the model's training distribution reflects promotion-month behaviour more proportionally.

3. **Use promotion-only training data for the recommendation task.** Since the goal is to choose *between* five promotions, rows with no promotion are arguably irrelevant to the ranking objective. Training exclusively on promotion months, with the baseline handled by the separate model, eliminates the imbalance problem entirely.

4. **Validate on a promotion-only held-out set** to ensure evaluation metrics reflect the model's ability to rank promotions, not its ability to predict baseline demand.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics

**Why random split is inappropriate**

The data has a temporal structure: each month's sales depend on prior months (seasonal patterns, rolling averages, lagged features). A random split allows future months' data to appear in the training set, leaking temporal information and inflating apparent model performance. The model would be evaluated on patterns it has effectively already seen, producing over-optimistic metrics that do not reflect real-world deployment performance where the model always predicts future months from past data.

**Correct approach: time-based split**

With 36 months of data across 50 stores:

- **Training set:** Months 1–24 (years 1–2), all 50 stores
- **Validation set:** Months 25–30 (first half of year 3) — used for hyperparameter tuning and feature selection
- **Test set:** Months 31–36 (second half of year 3) — held out until final evaluation

This mirrors real deployment: the model is trained on all available history and then predicts the next unseen month. For model selection, a **walk-forward cross-validation** scheme is preferred — train on months 1–12, validate on 13; train on 1–13, validate on 14; and so on — which produces multiple out-of-sample estimates and more robust model selection.

**Evaluation metrics**

| Metric | Formula | Business interpretation |
|---|---|---|
| RMSE (Root Mean Squared Error) | √(mean((ŷ − y)²)) | Measures typical prediction error in units of items sold; penalises large errors heavily. A RMSE of 50 means recommendations are off by ~50 items on average, which quantifies planning risk. |
| MAE (Mean Absolute Error) | mean(\|ŷ − y\|) | More robust to outlier stores; easier to explain to non-technical stakeholders as "on average, our predictions are off by X items." |
| Promotion Ranking Accuracy | % of store-months where the true best promotion matches the model's top recommendation | Directly measures the recommendation objective. A model with low RMSE but poor ranking accuracy fails the business task. |
| Spearman Rank Correlation | Correlation between predicted and actual rank order of promotions per store-month | Assesses whether the model correctly orders all five promotions, not just the top one. |

The primary metric for business decisions is Promotion Ranking Accuracy; RMSE and MAE are diagnostic metrics for improving the underlying volume predictions.

---

### B3(b) — Feature Importance for Explaining Different Recommendations

**Why the model recommends differently for the same store in different months**

The model recommends Loyalty Points Bonus in December and Flat Discount in March for Store 12. This reflects that, while store-level features are constant, temporal features (month, season, festival flags) and historical rolling features (prior-month items sold) change between months. The recommendation changes because the feature vector changes.

**Investigation process using feature importance**

1. **Global feature importance (SHAP summary plot):** Run SHAP (SHapley Additive exPlanations) on the full model to identify which features drive predictions most across all store-months. This establishes the baseline narrative — e.g., "month-of-year, promotion type, and footfall are the three most influential features globally."

2. **Local SHAP explanations for Store 12 in December vs March:** Compute SHAP values for the specific prediction rows — Store 12 × December × Loyalty Points Bonus and Store 12 × March × Flat Discount. A waterfall chart for each shows how each feature pushes the predicted items sold above or below the base rate.

   - For December: features like `is_festival_month = 1`, `season = winter`, `month = 12`, and high `rolling_avg_items_sold` likely push Loyalty Points Bonus prediction high, reflecting that customers in peak season respond to reward accumulation.
   - For March: features like `is_festival_month = 0`, `prior_month_items_sold` being lower (post-festival dip), and lower footfall likely favour Flat Discount, which is better at driving impulse purchases in low-traffic periods.

3. **Counterfactual comparison:** For each month, show the model's predicted items sold for all five promotions side by side. This helps the marketing team see not just the winner but the margin — "Loyalty Points Bonus is predicted to outsell Flat Discount by 120 items in December, but the gap shrinks to 15 in March, where Flat Discount takes the lead."

**Communication to the marketing team**

Present a simple one-page visual per store-month showing: (a) the recommended promotion, (b) predicted items sold for all five options as a bar chart, and (c) the top 3 SHAP drivers in plain language: "December's recommendation is driven by festival season and high expected footfall, which historically correlate with stronger loyalty reward redemption."

---

### B3(c) — End-to-End Deployment Process

**Step 1 — Saving the trained model**

After training and validation:
- Serialise the model artefact using `joblib` (scikit-learn) or the native format (e.g., `model.save()` for XGBoost/LightGBM).
- Package it alongside the preprocessing pipeline (scaler, encoder, feature list) and the store-segment assignment table into a versioned artefact bundle (e.g., `model_v1_2026_03.pkl`).
- Store in a model registry (MLflow, AWS S3 with versioning, or equivalent) with metadata: training date range, validation metrics, feature list, and model version.

**Step 2 — Preparing and feeding new monthly data**

At the start of each month (e.g., 1st of the month, before the marketing team's deployment decision):

1. **Data ingestion pipeline** pulls the previous month's transactions, the current month's calendar flags, and the current store attribute snapshot from the data warehouse.
2. **Feature engineering script** (identical to training pipeline) computes rolling averages, lag features, and calendar features for the current month.
3. **Scoring:** For each of the 50 stores, five rows are created (one per promotion type) with the computed features. The saved model scores all 250 rows in a single batch inference call.
4. **Recommendation generation:** For each store, select the promotion type with the highest predicted items sold. Output a recommendation table (store_id, recommended_promotion, predicted_items_sold, confidence_margin) to the marketing dashboard.
5. The entire pipeline is automated via a scheduled job (e.g., Airflow DAG or cron job) that runs on the 1st of each month without human intervention.

**Step 3 — Monitoring and retraining triggers**

| Monitor | What to track | Alert threshold |
|---|---|---|
| Prediction accuracy drift | Compare actual items sold (available mid-month) to predicted items sold from the start of the month | RMSE increases >20% above validation baseline over a 3-month rolling window |
| Feature distribution drift | Track mean and std of key input features (footfall, prior-month items) month-over-month using PSI (Population Stability Index) | PSI > 0.2 for any top-5 feature triggers investigation |
| Recommendation distribution | Track the share of each promotion type recommended across all stores each month | If one promotion is recommended for >70% of stores (collapse to one recommendation), the model may have overfit to a recent signal |
| Business outcome alignment | Track whether stores following the model's recommendation outperform stores where the marketing team overrode it | Ongoing A/B comparison; if override stores consistently outperform, model validity is questionable |

**Retraining schedule:**
- **Scheduled retraining:** Every 6 months with the full historical dataset, regardless of drift, to incorporate new data.
- **Triggered retraining:** Immediately if any monitoring alert fires — particularly if a new promotion type is introduced or if a structural event (e.g., post-pandemic consumer shift, major competitor entry) is detected in feature distributions.
- **Retraining process:** Re-run the full training pipeline (feature engineering → segment clustering → model training → walk-forward validation → metric comparison against current production model). Deploy the new model only if it improves on the held-out recent period; otherwise, keep the current model and investigate the root cause.
