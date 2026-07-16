"""Risk x Impact scoring - the business deliverable.

Combines the calibrated risk P(late) from the chosen model with a deterministic impact weight
(client importance, priority, parcel value) to rank each delivery for the dispatch queue.
"""
import matplotlib.pyplot as plt
from .feature_engineer import compute_impact_weight


def risk_impact_scores(best_model, best_name, X_test, deliveries, cfg, art, proba=None):
    """Score deliveries by risk x impact and save the ranked queue, scored table and a plot.

    Risk is the calibrated P(late); impact is the deterministic weight from client frequency,
    priority and parcel value. The product ranks the queue dispatch should attend to first.
    Parameters: best_model (calibrated); best_name; X_test; deliveries (ids/segments); cfg; art;
    proba optional precomputed P(late) for X_test.
    Returns: the scored DataFrame sorted by risk_x_impact (descending).
    """
    cols = ["delivery_id", "branch", "delivery_priority", "parcel_category", "parcel_value_sgd",
            "client_id_frequency", "delivery_priority_rank"]
    scored = deliveries.loc[X_test.index, cols].copy()
    scored["risk"] = proba if proba is not None else best_model.predict_proba(X_test)[:, 1]
    scored["impact"] = compute_impact_weight(scored, tuple(cfg["impact_weights"])).values
    # risk x impact: rank by likely-late AND costly-if-late, so dispatch attends to both
    scored["risk_x_impact"] = (scored["risk"] * scored["impact"]).round(4)
    scored = scored.sort_values("risk_x_impact", ascending=False)
    art.table(scored, "risk_impact_scored", index=False)
    art.table(scored.head(20), "top20_dispatch_queue", index=False)
    art.logger.info(f"Risk x impact scored {len(scored):,} deliveries with model '{best_name}'")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc = ax[0].scatter(scored["risk"], scored["impact"], c=scored["risk_x_impact"],
                       cmap="viridis", s=8, alpha=0.5)
    ax[0].set_xlabel("risk = P(late)"); ax[0].set_ylabel("impact weight")
    ax[0].set_title("Risk vs Impact (colour = score)"); fig.colorbar(sc, ax=ax[0])
    ax[1].hist(scored["risk_x_impact"], bins=40, color="#C44E52")
    ax[1].set_title("Distribution of risk x impact"); ax[1].set_xlabel("risk x impact")
    plt.tight_layout(); art.fig(fig, "risk_impact"); plt.close(fig)
    return scored
