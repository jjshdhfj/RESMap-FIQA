# RESMap-FIQA: publication rights and dependency review

Review date: 2026-11-24. **Status: NOT CLEARED for public redistribution.**
This is a technical rights checklist, not a legal opinion. It documents findings
from upstream source pages; it does not authorize redistribution of any asset.

| Component | Confirmed source evidence | Public release status |
|---|---|---|
| GFENet source (`resmap_generator/models/`) | Upstream https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement is public; its root file listing has no identifiable LICENSE, and its README does not clearly grant redistribution rights. | **BLOCKED pending a clear license or author permission**. GitHub visibility alone is not a sublicense. Do not automatically declare this code MIT/GPL. |
| GFENet pretrained checkpoint (`latest_net_G.pth`) | Upstream README links a downloadable GFE-Net checkpoint. | **Unresolved redistribution terms**. Keep checkpoint outside archive; link to upstream official download subject to its conditions. |
| RETFound implementation (`models_vit.py`, etc.) | Official RETFound repository (https://github.com/rmaphoh/RETFound) has an Attribution-NonCommercial 4.0 LICENSE. | Attribute upstream and comply with CC BY-NC 4.0; commercial use not allowed under those terms. Check provenance of each copied file. |
| RETFound pretrained CFP weights | Official https://huggingface.co/YukunZhou/RETFound_mae_natureCFP labels weights **CC BY-NC 4.0** and gates file access through acceptance. | **Do not mirror or redistribute**; tell users to obtain access from the publisher. |
| RGB examples (10) | Dataset appears to use EyeQ/EyePACS-derived image sources, but provenance of each of these 10 files was not independently established from the release itself. EyeQ README says its **code** is CC BY-NC-SA 4.0 and directs users to EyePACS for the original images. The original Kaggle EyePACS competition rules explicitly prohibit transmitting/publishing/redistributing competition data to nonparticipants. Whether each of these 10 images is subject to those terms must be verified before publication. | **BLOCKED pending proof of image reuse/redistribution permission for each sample**. EyeQ code license does not itself grant rights to EyePACS image content. |
| FOV masks + RESMaps | Derived from corresponding RGB image content and GFENet/U-head inference. | Treat as **unresolved** until underlying images and derivative redistribution permissions are checked. |
| U-head weights and A3 fine-tuned weights | Locally trained products. | Ownership/author/institution agreements and dependency license obligations must still be checked. |
| New portable launcher and audit scripts | Newly authored project utilities. | Can be separately released after upstream and institution review; do not assert one blanket license over all third-party code/assets. |

## Actions before public GitHub release

1. Obtain/review an explicit GFENet upstream license or written redistribution permission (source and checkpoint separately).
2. Confirm **each** sample's origin and whether EyePACS or any other source permits sharing processed RGB images, masks, or RESMaps. If uncertain, remove all ten image/mask/RESMap triplets from the **public** distribution and provide an authorized retrieval/preprocessing recipe instead.
3. Use the publisher's gated download for RETFound weights; do not place them in a ZIP or GitHub release.
4. Include appropriate third-party notices and original license text/links for any code actually redistributed, with changes marked.
5. Confirm U-head/A3 checkpoint ownership and whether university/research collaborators require approvals.
6. Perform a clean-environment install and full train-start smoke test; this is a technical requirement, not a copyright clearance.

## Sources reviewed

- GFENet repository: https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement
- RETFound repository/LICENSE: https://github.com/rmaphoh/RETFound/blob/main/LICENSE
- RETFound CFP weight card: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
- EyeQ dataset README: https://github.com/HzFu/EyeQ
- Kaggle EyePACS competition rules: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules
- GitHub guide on licenses: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-a-repository/licensing-a-repository

**Gate decision:** internal reproducibility review package permitted for local use; **do not upload this composite package publicly yet**.
