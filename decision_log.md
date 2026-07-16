# AIAP Technical Assessment — Decision Log

**Candidate name (as in NRIC):** Maanoj Shara Chandran

**Email (as used in your application):** maanojgshara@hotmail.com

---

## A note on this document

This decision log is the primary instrument by which we understand your thinking. The questions below cover the reasoning behind your work — from how you define the problem at the start to the decisions you made during the work itself. Do answer all five questions in your own words. 

You may use AI assistance freely on the technical deliverables (the EDA and the ML pipeline), but this Decision Log itself should be written by you — it is the record of your own thinking that we cross-check against your chat history.


---

## 1. Clarifying questions

What questions would you ask to better define and narrow the problem statement? For each question, briefly explain how the answer would meaningfully change your approach. 

Note: If it helps your decision-making, you may assume and list out the stakeholders' likely answers.


**Your answer:**

### Question 1
- What defines or counts as "trouble"?
- It is mentioned that customer complaints mainly mention late dleivereies, rouhgh handling of parcels and driving heivour.
- Only one of these is clearly on the data record - whether a delivery is late or not.
- Rough handling and driver behaviour  can only be inferred from the text in the feedback. Not all deliveries have feedback.
- Based on the stakeholder's answer, I can determine wheter my target varaible is clean and avlaiable in the dataset (should be close to 100%) or my target variable is sparse and could be baiased from the raitings/comment texts. If latness is the most concerning, my target varaible will be is_late to tbe derived from delviery_datetime and promised_delviery_datetime. Feedback ratings/comments can act as a validation role

*Assumed Stakeholder's Likely Answer*
  -  **"The major e-commerce client has mostly compalined about the lateness and it is something we can act on (meeting the deadlines etc)"**


### Question 2
- At what point in the delivery's lifecycle should the prediction be made?
- You mentioned "for each upcoming delivery" — but upcoming from when? At booking time, or at pickup time? This matters because it decides which columns are legitimately available as features and which would be leakage.
- At booking time, route/driver fields like driver_id, vehicle_type, and num_stops_on_route don't exist yet (no route or driver assigned). By pickup, they do — a parcel can't leave the depot without an assigned driver, vehicle, and planned route. Note also that the dataset has no explicit "dispatch" or "route-assigned" timestamp, so I have to tie the conceptual prediction point to a real column.
- delivery_datetime and anything derived from it,  can never be a feature but it actually determines the target
- Based on the stakeholder's answer, I can fix my prediction point and therefore my leakage-safe feature boundary. If they review the queue once routes/drivers are assigned (just before parcels go out), I map that to pickup_datetime as the prediction point 
  - which is the latest timestamp at which all operational features exist and no outcome has occurred. 
  - This makes all route/driver features and  promised_delivery_datetime  usable

*Assumed Stakeholder's Likely Answer*
  - **"Dispatch reviews the day's queue once routes and drivers are assigned, just before the parcels go out."**

### Question 3
- What do you mean by "how bad the trouble would be," and is that a number you want predicted, or a way to prioritise which problems to handle first?
- The possible measure for severity are: 
  - magnitude of lateness (minutes over the promise) 
  - the parcel's declared value (parcel_value_sgd)
  - the importance of the order (client_id and delivery_priority tier). 
- The Head of Operations asked for both probability ("how likely") and severity ("how bad")
  - which means i want something dispatch can see and act on with a fleet they can't expand
- Based on the stakeholder's answer, I can decide whether "how bad" requires a second predictive model or can be handled as a business rule on top of my classifier. 
  - If they want an actual predicted magnitude (e.g. expected minutes late), that implies a separate regression model. 
  - If they instead want to prioritise which predicted-late deliveries to act on first, I can keep a single classification model that outputs the probability of being late, P(late), and layer a deterministic impact weight on top, producing a risk × impact score. This choice also means the output probability from the model must be calibrated (to ensure a realisble score) 
    - The threshold will convert the probabilty into a flag, and the impact weight defines the triage ranking the dispatch team uses to focus their  attention on. 


*Assumed Stakeholder's Likely Answer*
  - **"We can't check every delivery. We want to focus on the ones most likely to be late. AND among those, which are the ones that will hurt us most (our major recurring clients and our Premium/VIP orders). We don't need an exact number of minutes. We need a ranked shortlist we can act on each morning"**
---

## 2. Defining the Problem Statement

Restate, in your own words, the refined problem you decided to solve. List your key assumptions. Briefly note what other framings you considered, and what you deliberately left out or scoped down, and why.


**Your answer:**

### Refined problem

- Build a supervised binary classification model that, at the point a parcel is picked up from the depot, predicts the probability that the delivery will miss its promised delivery time (is_late = delivery_datetime > promised_delivery_datetime). 
  - Pickup will be used as the prediction point because the dataset has no explicit dispatch assignment timestamp, and pickup_datetime is the latest moment at which all operational features (driver, vehicle, route stops) already exist and the  outcome has not occurred.
  - 3 classifiers will be trained and compared and evaluated 
    - The model will output P(late) , (a calibrated probablity) for each delivery
    - After probability calibration, an operating threshold will be chosen
    - Feature importance, SHAP analysis and error analysis (sliced by operational segment) will be also performed

- To address the "how bad" part, a deterministic impact weight will be layered 
  - This can be built (on top of the probability) from features (client_id proportion, delivery_priority, parcel_value_sgd etc) or derived features from the dataset 
  
- Output a risk × impact score. 
  - risk = P(late),       Impact = impact weight
  - This score can point out the deliveries that are both likely to fail and costly if they do fail
  - It is what the dispatch team can use each morning to rank the day's queue and focus their limited attention on the deliveries that are both most likely to be late and most costly if they are. 
  - This will keep the solution to a single classification target with a business ranking rule
  - Effort is reprioritzed, isntead of adding more drivers
    




### Key assumptions

- promised_delivery_datetime is treated as a hard deadline 
- Client importance is proportionate to how often the client_id appears in the dataset
- The label is binary is_late and can be derived from avialble features. 
- The prediction is made at pickup. 
  - Features known at or before pickup are allowed; 
  - delivery_datetime and all feedback fields are excluded to prevent leakage. 
- Customer ratings/comments are not the target. It may be used to validate that lateness tracks dissatisfaction.
- Impact weight can be calcuated/derived from features in the dataset 
- The historical data is representative of near-future operations, since the same  fleet of drivers are used. 
- A missed late delivery (false negative) is generally more costly than a false alarm (false positive), especially for major clients and high tiered orders

### Other framings considered and scoped down

- Predicting the customer rating directly 
  - Rejected as the primary target because not all deliveries may be rated. Signal may be sparse.
- NLP on comments to model rough handling / driver behaviour 
  - These may be real complaints but might have low coverage and carry noisy signal, and are hard to act on before delivery.
- Regressing the magnitude of lateness (minutes over). 
  - Considered for the "how bad" part, but it would reqire a second regression model. 
  - "how bad" part can be handled as a deterministic impact weight on the classifier output, keeping the scope to one classification problem.
- A diagnostic study of why ratings fell. 
  - I might be able to recover much of this insight via feature importance and SHAP from the predictive models.



---

## 3. Key decisions during Solution Development

Walk through three key decisions you made during Solution Development. For each: what options did you consider, what did you choose, and why? These can be technical (modelling choices, feature handling, evaluation metrics) or about the work itself (what to prioritise, what to drop, how to spend your time).


**Your answer:**

### I chose to deduplicate the two tables before joining them, after verifying the duplicates were safe to drop
- During EDA I had to decide the order of two operations 
  - join the delivery and feedback tables on delivery_id first and then deduplicate 
  - deduplicate each table individually and then join.
- I considered that joining first could multiply row, making it harder to tell real duplicates from join artefacts.
- Before dropping anything, I verified it was safe. I checked that rows duplicated on delivery_id were exact full-row duplicates.
- I then deduplicated each table on its own first, then joined on delivery_id.

### I chose to use features which are known only at pickup-time which will be a fair experiment
- I considered 2 options
  - Use all duration / feedback features including pickup_to_delivery_mins, booking_to_delivery_mins and delivery_datetime 
  - Use features which are only known at pickup. (CHOSEN)
- The Head of ops wants something that helps dipatch team 'see'. 
- Means the model is meant to run at dispatch. 
- So any feature containing the outcome is leakage and would not exist at prediction time.  
- Feedback features also cannot be inlcdued during training as feedback occurs after delviery time. 
- So any features including those derived must be used from featuresw which exist at or before pickup time



### I chose to use evaluation metrices which are cost aware and account for the heavy target imbalance.  
- I considered 2 options
  - use accuracy as a primary metric of evaluation at a default 0.5 threshold
  - use PR-AUC as the primary metric of evaluation (recall and brier as metrices to break ties) with calibration and a cost-aware cut. (CHOSEN)
- DUe to the 11% positive rate, accuracy will be meaningless. 
- Calibration had to be done because P(late) is multiplied by the impact weight and if the prob is not calibrated , the ranking would be meaningless 
  - which was useful because it improved Brier  from ~0.20 to ~0.09 after calibration was done  
- i also chose the 5:1 cost ratio to address the concern that missing a late delivery is worse than a false alarm. 



---

## 4. Use of the AI assistant

Where did you use the AI assistant in this work? Give three specific examples of something the assistant suggested that you changed, rejected, or significantly modified — and explain your reasoning.

**Your answer:**

I used the assistant throughout the EDA and the ML pipeline, but I made the framing, data-handling and modelling judgement calls myself. Three specific cases where I changed, rejected, or modified its suggestions:

### 1. I modified the AIs categorical preprocessing after its code crashed on missing values
- The AI gave me a preprocessor which fed the nominal/high-cardinality columns straight into the Encoders without imputation. 
- When I ran it, it threw "TypeError: boolean value of NA is ambiguous".
- The columns were pandas string dtype, which uses pd.NA and the encoder can't sort pd.NA mixed with strings (Something i learned this time)
- The AI offered a quick fix (cast to object dtype so np.nan is procesasble) but I rejected the quick hack and instead added a categorical imputer inside the ColumnTransformer, so missing-value handling is refit per CV fold and applied identically to test/production data. 
- This was the more leakage-safe  fix.

### 2. I rejected SMOTE as the first-line imbalance fix in favour of class weights
- When handling the  class imbalance, the assistant presented resampling alongside class weights as options.
  - I chose class_weight to be balanced at first 
  - SMOTE will be done if minority recall stayed too low.
  - Class weights add no synthetic data and no leakage risk, whereas SMOTE must be fit strictly inside CV folds to avoid leakage
  - Class weights are risk-free but SMOTE is risky and must be done properly or else there is a risk of leakage

### 3. I modified how empty/whitespace-only comments were handled to preserve the signal
- I asked whether to fill blank or whitespace-only comments with a placeholder like "No comment".
- The AI advised against a plain placeholder and to treat them as missing for analysis and add a separate indicator flag so the fact that a customer left no text isn't lost.
  - I chose this over the simpler laceholder string  fill, because a placeholder string into the comment field could hide a potentially meaningful signal (whether feedback was left at all)
- This can keep missingness explicit and recoverable

---

## 5. Next Steps

If you had one more week to continue this project, what would you do next, and why? What signals from your current work make those the right next steps?


**Your answer:**


1) I would validate the impact weighting and back-test the risk × impact ranking, since it is currently an unvalidated business rule
    - The impact weight is a hand-set blend (0.4/0.4/0.2 over client frequency, priority tier and log parcel value)
    - Also client importance is only guessed by how often a client_id appears 
    - So i would set the weights with operations (what a late VIP for the major client truly costs vs a late standard parcel)
    - THen i would prolly sensitivity-test that the top-N ranking is stable to the exact weights
    - Finally i will back-test whether the top-ranked queue actually captures the costly late deliveries on held-out data.
 

2) I would try to boost the model's recall on the highest-impact segments at the source and not just at the threshold
    - The bias slices show the model misses exactly the deliveries the business cares most about 
      - Recall is only 0.067 (VIP), 0.173 (2-hour), and 0.186 (Premium) 
      - FN rates are above 0.8 in those groups.
    - My per-segment thresholds are currently near the global cut and don't help 
        - A 0.01 shift can't fix a 0.067 recall. The threshold only changes the at-risk flag, not the (risk × impact) ranking 
    - So i would improve recall during training by maybe sample weighting toward high-impact deliveries 

3) I would tackle the other 2 complaints meantioned - rough handling of parcels    and      driver behvaoiur
    - my model only addreseds the late delivery problem and not the other 2 issues
      - if i can address the other 2 issues, i will be able to satisfy the business requiremetts 
      - I could perfrom Natrural lang processing comments field on the  deliveries which had comments filled
          - this can help to  surface parcel handling and driving behaviour signals
          - i can use these sigansls to see if they caused the ratings to drop

---