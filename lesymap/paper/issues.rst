

----------------------------------------------------------------------------

*Two reviewers noted that **the experimental results did not fully support the conclusion** that associative models were sufficient for lesion-behavior mapping. It seems that the **causal model performs better in some cases and worse in others**. One reviewer found serious weaknesses with the premise of the paper, including the, **lack of a clearly defined modeling goal** a **potentially biased data generation process**, and an **unclear causal modeling approach**. This reviewer makes a good case that the conclusions are overstated. The authors should clearly address these concerns in their rebuttal.*



*The apparent argument is that associative models can suffice for the problem mapping behavioural deficits to brain lesions when contrasted against causal models. Regardless of whether one favours one type of model or the other, it is the supposed antagonism which is unclear to me. From a mathematical point of view, associative and causal models addressed different problems, they have different assumptions, different outcomes, etc. From a strictly mathematical point of view, they are not antagonists! So unless some mathematical proof is given to me that this is the case, then I do not see much the point of the authors. If a mere associative relation is sought, then associative methods should be used. If a causal statement is intended, then a causal should be employed. Now, the authors never disclose a formal problem statement, so it is difficult to assess whether they have in mind an associative problem, or a causal problem.*

*The causal framework is unclear. At the beginning one may think the authors opted for Pearl’s causality, but then the double robustness method which is based in Rubin’s causality is used. Also, causal assumptions (causal sufficiency, faithfulness, etc) are unclear, the structure learning algorithms are unclear, etc*

*The forward linear model to generate the synthetic behavioural outcomes does not appear to have a physiological basis or any other physical rationale. If it is simply an associative model, why should a causal model later out-performed a simple associative model?*

*The departing criticism for neglecting univariate analysis is the there may be correlates among regions. True, but then there will be the analogous criticism neglecting for multivariate analysis, e.g. existence of explicative latent variables. Perhaps, this is my bias, but my intuition from mathematics is aligned with the idea of no free lunch, so if one intends to make an argument that one type of model is better or sufficient (as in this case) for a certain problem, it must be aware that there will be scenarios where other models challenge the supremacy of the claimed method.*

*+ The interpretation of Table 1 is misguided. The associative models are oriented to maximize explained variance, whereas causal models aren’t.*

*+ Inferential statistics are missing.*


*+ BART is hardly state-of-the-art anymore.*

*+ LASSO and variants regularize using the l1 norm but I have no mathematical guarantees, only sparse inconclusive empirical evidence, that this may be the adequate regularization for the problem at hand.*


