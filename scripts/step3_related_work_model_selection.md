# 3. Related Work and Model Selection

## Related Work

Mobile/cellular traffic forecasting on this exact Milan dataset (Telecom
Italia Big Data Challenge) has been studied extensively since its release,
giving strong precedent for model choice. Barlacchi et al. (2015) is the
original dataset paper; subsequent work has applied recurrent, convolutional,
and tree-based methods to it.

Trinh et al. compared LSTM and GRU on this same Milan dataset (two months of
data, same grid structure as ours) for short-term Internet traffic
forecasting, clustering cells by activity level beforehand. Their finding —
that LSTM effectively captured both within-day and across-two-month
seasonality, and generally outperformed GRU — is a direct precedent for our
first model choice.

Zhang et al. combined LSTM with Gaussian Process Regression specifically on
Milan cellular traffic, using a strategy of separating dominant periodic
components before feeding the residual to the LSTM — reinforcing that this
dataset has strong, learnable periodicity that recurrent architectures can
exploit.

Temporal Convolutional Networks (TCNs), introduced by Bai et al. (2018) as a
convolutional alternative to RNNs for sequence modelling, have since been
applied directly to cellular traffic prediction, including hybrid TCN-based
architectures evaluated on Milan traffic data. TCNs use dilated causal
convolutions to reach a large receptive field without the vanishing-gradient
issues RNNs can face, and support parallel computation during training (a
practical advantage given the dataset's size).

Tree-based gradient boosting methods (XGBoost, LightGBM) are also
established baselines in cellular/network traffic forecasting research on
this and similar datasets, typically framed as a regression problem over
lag and rolling-window features rather than as a native sequence model.
This reframing is itself a useful point of contrast with LSTM/TCN, since it
tests whether explicit recurrence or convolution over the raw sequence is
actually necessary, or whether hand-engineered lag/window features are
sufficient to capture the periodicity observed in Task 2.

## Model Selection and Justification

Three models were selected: **LSTM**, **TCN**, and **LightGBM**. These
represent three structurally distinct approaches — recurrent, convolutional,
and tree-based — satisfying the requirement for models that are more than
minor variations of each other, and allow the comparison to test different
inductive biases against the same data.

### LSTM (Long Short-Term Memory)

Justification: Task 2's ADF test on the top-traffic square (5161) produced a
statistic of -19.03 (p < 0.0001), rejecting the unit-root null far beyond
the 1% critical value (-3.43) — the series is stationary in the classical
sense, but the ACF/PACF and decomposition results show this stationarity
coexists with strong recurring daily structure (24-hour periodicity visible
in both). LSTM's gating mechanism is well-suited to this combination: it
can maintain information over the ~144 lags (10-minute intervals) needed to
span a full day, which is exactly the horizon our ACF/PACF analysis showed
carries predictive signal. This is also the architecture directly validated
on this same Milan dataset by prior work (Trinh et al.), giving a strong
empirical precedent rather than a purely theoretical one.

### TCN (Temporal Convolutional Network)

Justification: Our small-multiples comparison (fig2b) across five squares
(5161, 5059, 5259, 4159, 4556) showed not just differing magnitudes but
differing *shapes* of daily cycle — squares 5161/5059/5259 show sharp
business-hours peaks with troughs near zero, while 4556 maintains a much
higher overnight floor relative to its peak, suggesting a different, more
residential usage pattern. A TCN's dilated convolutions give it a large,
tunable receptive field while remaining computationally cheaper to train
than an RNN across many areas — relevant given we need to run experiments
across three separate geographical areas with potentially different
temporal profiles. TCNs also avoid the sequential (non-parallelizable)
computation of LSTMs, a practical benefit given the ~89 million rows
processed in Task 1.

### LightGBM (Gradient Boosted Trees)

Justification: Unlike LSTM and TCN, LightGBM has no innate
sequence-modelling mechanism — it requires the forecasting problem to be
reframed as tabular regression using explicit lag features (e.g.,
$x_{t-1}, x_{t-2}, ..., x_{t-144}$) and calendar/window features (hour of
day, day of week, rolling means). This is deliberately included as a
contrasting baseline: given that Task 2 found strong, regular 24-hour
periodicity (via ACF/PACF and decomposition), a well-engineered set of lag
features may let a tree-based model capture most of the same signal a deep
sequence model would learn implicitly — at a fraction of the training cost.
This directly tests whether the added complexity of LSTM/TCN is justified
by the data's actual characteristics, which is central to the assignment's
comparative objective rather than simply "finding the lowest error."

## References

[1] Barlacchi, G., De Nadai, M., Larcher, R., et al. (2015). A multi-source
dataset of urban life in the city of Milan and the Province of Trentino.
Scientific Data, 2, 150055. https://doi.org/10.1038/sdata.2015.55

[2] Trinh, H.D. et al. Predicting Short-term Mobile Internet Traffic from
Internet Activity using Recurrent Neural Networks. https://arxiv.org/pdf/2010.05741

[3] Zhang, C. et al. Cellular Traffic Load Prediction with LSTM and Gaussian
Process Regression. IEEE. https://ieeexplore.ieee.org/document/9148738/

[4] Bai, S., Kolter, J.Z., Koltun, V. (2018). An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling.

[5] Regional Correlation Aided Mobile Traffic Prediction with Spatiotemporal
Deep Learning (Multi TCN-LSTM). https://arxiv.org/pdf/2312.06279

[6] A novel hybrid framework based on temporal convolution network and
transformer for network traffic prediction (evaluated on Milan traffic
data). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10490908/