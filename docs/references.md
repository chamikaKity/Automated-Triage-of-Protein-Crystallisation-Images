# References

Published ideas used in this project, cited per CLAUDE.md rule 9 (originality - code is our
own, but techniques/methodology from published work are credited here).

- **Model randomisation sanity check for saliency maps:** Adebayo, J., Gilmer, J., Muelly, M.,
  Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity Checks for Saliency Maps.* NeurIPS 2018.
  Used in `notebooks/04_xai.ipynb`'s second sanity check: Grad-CAM should change meaningfully
  when the trained layers' weights are randomised, otherwise the explanation isn't actually
  reflecting what the model learned.
- **Grad-CAM:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
  (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
  ICCV 2017. The general method implemented from scratch in `src/marco/xai.py` (not copied
  from any existing implementation).
- **LIME:** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?":
  Explaining the Predictions of Any Classifier.* KDD 2016. Used via the `lime` package
  (`lime.lime_image`) in `notebooks/04_xai.ipynb`.
