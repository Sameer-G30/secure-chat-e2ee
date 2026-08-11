# Literature review: client-side phishing and scam detection for E2EE chat

## Scope and research questions

This review supports a chat system in which plaintext exists only on endpoints. It asks:

1. Which public corpora cover short informal spam, phishing language, and legitimate text?
2. How severe is the domain shift from email or SMS to two-person chat?
3. Which baseline and transformer methods are credible for text classification?
4. Which model-size and privacy constraints matter when inference runs in a browser?

Dataset sizes and licences below were checked against original project pages where possible on 3 August 2026. A dataset's hosting page and its underlying messages may have different rights; every source will be frozen with a retrieval date, checksum, and provenance record before training.

## Dataset survey

### 1. UCI SMS Spam Collection — primary short-text source

- **Source:** [UCI dataset 228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
- **Size and balance:** 5,574 English SMS messages: 4,827 ham (86.6%) and 747 spam (13.4%).
- **Licence:** CC BY 4.0 on the current UCI page.
- **Fit:** This is the closest public source to chat because messages are short, informal, and phone-oriented. It is still one-way SMS rather than a conversation and is old enough to underrepresent current investment, delivery, account-takeover, and crypto scams.
- **Use:** Main Slice 1 source and later baseline training source. The preparation script normalizes all 5,574 records and preserves the published 4,827/747 class counts.

### 2. Enron-Spam — broad legitimate/spam email source

- **Source:** the Enron-Spam derivative described by Metsis, Androutsopoulos, and Paliouras at CEAS 2006; the underlying legitimate corpus is documented by [CMU](https://www.cs.cmu.edu/~enron/).
- **Size and balance:** the common six-user Enron-Spam derivative contains 33,716 messages: 16,545 ham and 17,171 spam. This must not be confused with CMU's roughly 0.5-million-message Enron corpus.
- **Licence/status:** the FERC-released material is public research data, but CMU does not state a simple open-source licence and asks users to respect the privacy of people represented in the corpus. The exact derivative's redistribution terms must be verified before checking any copy into a release.
- **Fit:** It adds realistic business language and diverse spam but is much longer and more formal than chat. Headers, quoted replies, signatures, and boilerplate could become shortcuts unless removed or explicitly modeled.
- **Use:** Candidate supplementary training source, stored locally only until provenance and redistribution are confirmed.

### 3. Apache SpamAssassin Public Corpus — difficult legitimate-email examples

- **Source:** [Apache SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html).
- **Size and balance:** 6,047 messages total: 500 original spam, 2,500 easy ham, 250 hard ham, 1,400 easy-ham-2, and 1,397 spam-2; approximately 31% spam.
- **Licence/status:** the project describes the corpus as suitable for testing and its messages as public or submitted with publication knowledge, but copyright in message text remains with original senders. It is not equivalent to an Apache-2.0 code licence.
- **Fit:** `hard_ham` is useful for measuring false positives on promotional-looking legitimate text. Full email formatting is unlike chat and can leak source-specific header patterns.
- **Use:** Validation and supplementary training after body extraction, deduplication, and a written redistribution decision.

### 4. Nazario Phishing Corpus — phishing-specific malicious source

- **Source:** [Jose Nazario's original corpus](https://monkey.org/~jose/phishing/) and its [README](https://monkey.org/~jose/phishing/README.txt).
- **Size and balance:** an evolving phishing-only collection distributed in historical mbox files and annual batches; recent literature reports 11,527 samples for a selected snapshot. We will record the exact selected files and resulting count rather than treating that number as timeless.
- **Licence:** CC BY 4.0 according to the current original README.
- **Fit:** It contributes real credential-theft and impersonation language missing from generic spam. It has no legitimate class, is email-shaped, and later files may contain unredacted destination information.
- **Use:** Positive-class supplement after privacy review, body extraction, deduplication, and source-level split controls.

### 5. Kaggle “Phishing Emails” by `subhajournal` — compiled candidate

- **Source:** [exact Kaggle dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails), independently described by Uddin, Mahiuddin, and Sarker (2026).
- **Size and balance:** 18,650 rows before cleaning: 11,322 safe (60.7%) and 7,328 phishing (39.3%). Sixteen null bodies leave 18,634 usable rows: 11,322 safe and 7,312 phishing.
- **Licence/status:** Kaggle and mirrors label the compilation LGPL-3.0, but upstream message provenance and the uploader's authority to apply that licence are not sufficiently documented. Research use is plausible; redistribution and commercial use remain uncertain.
- **Fit:** It provides a convenient labeled email compilation, but aggregation may overlap Enron, SpamAssassin, Nazario, or other sources. Such overlap can inflate metrics if near-duplicates cross splits.
- **Use:** Candidate only. Before adoption we will inspect labels, trace upstream provenance, hash normalized texts against every other corpus, and reject it if provenance is insufficient.

### 6. Hand-curated chat-style evaluation set — deployment-domain check

- **Planned size and balance:** the first scoped release contains 300 short conversational messages, intentionally balanced at 150 scam and 150 legitimate examples. A stronger later benchmark should expand toward 2,400 examples (1,200 per class) if time and independent annotation capacity permit. A balanced diagnostic set does not estimate real-world prevalence, so production-like prevalence will be tested separately.
- **Structure:** half single-turn messages and half two-to-four-turn conversations, with paraphrases grouped by campaign so one campaign family cannot leak across development and evaluation.
- **Scam coverage:** credential theft, fake support, delivery/payment requests, job scams, investment/crypto scams, romance grooming, urgency and impersonation, malicious-link prompting, and gift-card requests.
- **Legitimate hard negatives:** genuine payment reminders, security discussions, harmless shortened links, urgent family coordination, ordinary marketplace conversations, and quoted scam-awareness text.
- **Licence:** repository-owned content under the repository licence. No private conversations will be copied. Any assisted drafts must be manually rewritten/reviewed and marked with provenance.
- **Fit:** This is the only set tailored to post-decryption chat inference.
- **Use:** Locked final evaluation only. It must never be included in training, feature design, or threshold tuning. At least two reviewers should label each example, with disagreements retained for analysis.

## Dataset risks and controls

- **Domain leakage:** Split by source and near-duplicate cluster, not by randomly shuffling rows alone.
- **Temporal leakage:** Keep a newer malicious-message slice for evaluation where collection dates permit.
- **Identity leakage:** Strip addresses, names, message IDs, headers, signatures, and quoted chains unless a feature is deliberately retained and available in chat.
- **Class imbalance:** Report class-specific precision, recall, and F1; do not use accuracy as the main metric.
- **Threshold selection:** Tune on a validation set, then freeze the threshold before opening the chat-style test labels.
- **Provenance:** Store source URL, retrieval time, checksum, licence note, transformations, and exclusions in a data manifest.
- **Privacy:** Do not commit raw email corpora. Review unredacted sources before local processing.
- **Reproducibility:** Download scripts should target original sources; manual Kaggle downloads should be checksum-pinned and documented.

## Research papers

### 1. Almeida, Gómez Hidalgo, and Yamakami (2011)

**“Contributions to the Study of SMS Spam Filtering: New Collection and Results.”** Proceedings of ACM DocEng 2011, pp. 259–262. DOI [10.1145/2034691.2034742](https://doi.org/10.1145/2034691.2034742).

- **Takeaway:** Introduced the public SMS collection and found SVM to be a strong baseline among established classifiers.
- **Use here:** Supports starting with a transparent TF-IDF linear model before a transformer.
- **Difference:** Our deployment text is interactive chat, our output is a warning score rather than a mail filter decision, and inference must run privately in a browser.

### 2. Delany, Buckley, and Greene (2012)

**“SMS Spam Filtering: Methods and Data.”** *Expert Systems with Applications*, 39(10), 9899–9908. DOI [10.1016/j.eswa.2012.02.053](https://doi.org/10.1016/j.eswa.2012.02.053).

- **Takeaway:** Compares SMS-specific representations and classifiers and emphasizes that short, informal text behaves differently from email.
- **Use here:** Motivates tokenization experiments, URL/number handling, and robust comparison rather than assuming email preprocessing transfers.
- **Difference:** We add phishing/scam subclasses, out-of-domain testing, probability calibration, and browser cost measurements.

### 3. Gómez Hidalgo, Almeida, and Yamakami (2012)

**“On the Validity of a New SMS Spam Collection.”** 11th IEEE International Conference on Machine Learning and Applications. DOI [10.1109/ICMLA.2012.211](https://doi.org/10.1109/ICMLA.2012.211).

- **Takeaway:** Investigates duplicate and source-combination concerns in the SMS corpus rather than accepting collection construction uncritically.
- **Use here:** Directly motivates deduplication before splitting and source-aware evaluation.
- **Difference:** Our combined corpus has an even larger leakage risk because it merges SMS, email, phishing-only, and curated chat sources.

### 4. Fang, Zhang, Huang, Liu, and Yang (2019)

**“Phishing Email Detection Using Improved RCNN Model With Multilevel Vectors and Attention Mechanism.”** *IEEE Access*, 7, 56329–56340. DOI [10.1109/ACCESS.2019.2913705](https://doi.org/10.1109/ACCESS.2019.2913705).

- **Takeaway:** THEMIS combines header/body and character/word representations, reporting a low false-positive rate on an imbalanced phishing dataset.
- **Use here:** Reinforces measuring false positives and considering character-level robustness to obfuscation.
- **Difference:** Chat lacks email headers, and an RCNN with multiple representations is less attractive for a small WebAssembly payload than a linear baseline or quantized compact transformer.

### 5. Meléndez, Ptaszynski, and Masui (2024)

**“Comparative Investigation of Traditional Machine-Learning Models and Transformer Models for Phishing Email Detection.”** *Electronics*, 13(24), 4877. DOI [10.3390/electronics13244877](https://doi.org/10.3390/electronics13244877).

- **Takeaway:** DistilBERT, BERT, and RoBERTa outperformed traditional models on their compiled phishing data, while SVM remained a strong lightweight comparator.
- **Use here:** Justifies retaining both a linear baseline and a transformer upgrade, then comparing more than predictive scores.
- **Difference:** We must include model download size, cold-start latency, memory, and chat-domain F1; a small metric gain may not justify endpoint cost.

### 6. Thapa et al. (2023)

**“Evaluation of Federated Learning in Phishing Email Detection.”** *Sensors*, 23(9), 4346. DOI [10.3390/s23094346](https://doi.org/10.3390/s23094346).

- **Takeaway:** Evaluates phishing models under federated learning so raw organizational email does not need to be centralized, while showing that client data distribution affects results.
- **Use here:** Provides a privacy-preserving comparison point and highlights non-IID/domain-shift risk.
- **Difference:** This project performs only inference on-device; training remains offline on public data, so no message-derived gradients leave the browser.

### 7. Sanh, Debut, Chaumond, and Wolf (2019)

**“DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter.”** arXiv [1910.01108](https://arxiv.org/abs/1910.01108).

- **Takeaway:** Knowledge distillation reduced BERT size by about 40%, retained about 97% of reported language-understanding capability, and improved inference speed by about 60%.
- **Use here:** Supports DistilBERT as the planned transformer candidate rather than full BERT.
- **Difference:** This is a general model paper and preprint, not a phishing or browser-WASM evaluation. Our project must measure actual ONNX Runtime Web behavior and quantized weight size.

### 8. Sun et al. (2020)

**“MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices.”** Proceedings of ACL 2020, pp. 2158–2170. DOI [10.18653/v1/2020.acl-main.195](https://doi.org/10.18653/v1/2020.acl-main.195).

- **Takeaway:** A thin, deep bottleneck architecture achieved a reported 4.3× size reduction and 5.5× speedup over BERT-base while remaining competitive on benchmark tasks.
- **Use here:** Establishes a credible alternative if DistilBERT does not meet browser latency or memory targets.
- **Difference:** Native mobile benchmarks do not establish WebAssembly performance, and MobileBERT export/tokenizer complexity must be tested separately.

### 9. Jiao et al. (2020)

**“TinyBERT: Distilling BERT for Natural Language Understanding.”** Findings of EMNLP 2020, pp. 4163–4174. DOI [10.18653/v1/2020.findings-emnlp.372](https://doi.org/10.18653/v1/2020.findings-emnlp.372).

- **Takeaway:** Two-stage transformer distillation produced a four-layer model reported as 7.5× smaller and 9.4× faster than BERT-base while retaining 96.8% of teacher performance on GLUE.
- **Use here:** Provides a fallback architecture and demonstrates that task-specific distillation can matter.
- **Difference:** Training a distilled student adds project complexity; it is an optimization path only if the simpler baseline and off-the-shelf DistilBERT fail measured UX targets.

### 10. Cohn-Gordon, Cremers, Dowling, Garratt, and Stebila (2017)

**“A Formal Security Analysis of the Signal Messaging Protocol.”** IEEE European Symposium on Security and Privacy. DOI [10.1109/EuroSP.2017.27](https://doi.org/10.1109/EuroSP.2017.27).

- **Takeaway:** Formally analyzes Signal's key agreement and ratcheting security, including post-compromise properties.
- **Use here:** Provides the standard against which this project's deliberately simpler static-X25519 and epoch-separation design must be described.
- **Difference:** We do not claim Signal-equivalent forward secrecy or post-compromise security; a complete Double Ratchet remains out of scope.

### 11. Reaves, Blue, Tian, Traynor, and Butler (2016)

**“Detecting SMS Spam in the Age of Legitimate Bulk Messaging.”** ACM WiSec 2016. DOI [10.1145/2939918.2939937](https://doi.org/10.1145/2939918.2939937).

- **Takeaway:** Earlier classifiers degraded sharply when legitimate bulk messages were introduced, showing that easy ham produces misleading results.
- **Use here:** Supports hard-negative chat examples, temporal evaluation, and explicit benign false-positive reporting.
- **Difference:** Our warning system must also handle multi-turn scams and modern transactional messages after local decryption.

### 12. Otieno, Namin, and Jones (2023)

**“The Application of the BERT Transformer Model for Phishing Email Classification.”** IEEE COMPSAC 2023. DOI [10.1109/COMPSAC57700.2023.00198](https://doi.org/10.1109/COMPSAC57700.2023.00198).

- **Takeaway:** Reports a strong BERT phishing baseline on Nazario-positive and Enron-negative data.
- **Use here:** Confirms transformer viability while exposing the need to test whether performance survives source-held-out and chat-domain evaluation.
- **Difference:** Source and label are tightly coupled in that experimental setup, and full BERT is too costly for our default browser path.

### 13. Uddin, Mahiuddin, and Sarker (2026)

**“An Explainable Transformer-Based Model for Phishing Email Detection: A Large Language Model Approach.”** *Computer Networks*, 2026, 112061. DOI [10.1016/j.comnet.2026.112061](https://doi.org/10.1016/j.comnet.2026.112061).

- **Takeaway:** Fine-tunes RoBERTa on the identified Kaggle corpus and combines explanation methods for user-facing interpretation.
- **Use here:** Supports transparent warning UX and provides independently published details about the candidate dataset.
- **Difference:** Its oversampled email experiment does not measure campaign-held-out chat performance or browser inference cost.

### 14. Cormack, Gómez Hidalgo, and Puertas Sánz (2007)

**“Spam Filtering for Short Messages.”** ACM CIKM 2007. DOI [10.1145/1321440.1321486](https://doi.org/10.1145/1321440.1321486).

- **Takeaway:** Demonstrates that short-message filtering benefits from representations adapted to compact text.
- **Use here:** Supports keeping character and tokenization experiments alongside the TF-IDF baseline.
- **Difference:** The study predates current phishing tactics, conversation context, and client-side model constraints.

## Synthesis and resulting design

The evidence supports a staged model strategy:

1. **Baseline first:** word/character TF-IDF with Logistic Regression or Linear SVM, evaluated with stratified and source-held-out splits.
2. **Deployment-domain gate:** do not claim success unless the locked chat-style set has acceptable scam recall and warning precision.
3. **Transformer second:** fine-tune DistilBERT, then quantize and compare it with the baseline on F1, calibration, bytes downloaded, cold load, warm latency, and memory.
4. **Endpoint privacy:** export inference artifacts for ONNX Runtime Web; never send decrypted text, predictions, embeddings, or gradients to the server.
5. **User control:** display a non-blocking warning above the message instead of hiding or deleting content.

The expected research contribution is not a novel classifier architecture. It is an honest end-to-end engineering evaluation of how a useful scam detector can operate after local E2EE decryption without granting the relay server plaintext access.

## Slice 1–2 data status

- The UCI download/normalization script is implemented at `ml/scripts/download_sms_spam.py`.
- Enron-Spam, SpamAssassin, Nazario, and Kaggle phishing preparation scripts are implemented under `ml/scripts/`.
- Shared email body extraction lives at `ml/src/secure_chat_ml/email_text.py`.
- The normalized schema is defined at `ml/data/label-schema.yaml`.
- Provenance pins for all admitted corpora are recorded in `ml/data/sources.yaml`.
- The executed SMS EDA notebook is at `ml/notebooks/01_eda.ipynb`.
- The executed multi-corpus EDA notebook is at `ml/notebooks/02_eda_all_corpora.ipynb`.
- Raw and processed corpora are intentionally ignored by Git.
- Kaggle remains a candidate for combined training until overlap and upstream provenance controls are applied during split design.
