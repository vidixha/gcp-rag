# Prashant Khatri - Resume Information (Structured)

## Contact Information
- Email: prashantkhatri23@gmail.com
- Phone: 9784939728

## Education

| Degree | Institution | Specialization | CPI/% | Year |
|--------|-------------|----------------|-------|------|
| M.Tech | IIT Bombay | Electrical Engineering (Communication Engineering) | 8.74 | 2022-2024 |
| B.Tech | IIT Guwahati | Electronics and Communication Engineering | 8.43 | 2018-2022 |
| Class XII | - | - | 92% | 2018 |
| Class X | - | - | 94% | 2016 |

## Work Experience

**Machine Learning Engineer I -> II** | Hyprbots Systems (Fintech Startup) | Jul 2024 - Present (~1.5 years)

### Key Projects at Hyprbots:


#### 1. Document Splitter & Classifier [Ongoing]
"""
thoughts: 
1. The first thought is that we should maybe place this work above the OCR work because, though in some aspects of research are present in OCR (more of a document ingestion module that support most of the prevelant file type and ouputs them in standerdised format ready to be used by downstream modules ex for pdf and images it will ouput pagewise layout aware text and pagewise processed image in numpy format - after being read and angle (0,90,180,270) and skew corrected, for excel it will output worksheet wise image and  ), but it's more of a combination of engineering and less of research skills there highlighted. 
"""
- Built configurable document classifier switching between VLMs (single-page/image inputs) and LLMs (multi-page documents) based on input format
- Building Page Stream Segmentation (PSS) module for separating document bundles into coherent multi-page documents
- Developed evaluation framework with page-level, document-level, and stream-level metrics


#### 2. OCR & Document Ingestion Pipeline
"""
thoughts: 
1.  I think we should more responsibly present the information regarding this module: OCR is a major component of this module in terms of effort put on to implementing things but some time calling this module OCR can be confusing (Maybe we can come up with a better name which still shows the importance of the effort that has been put in the OCR, but does not confuse us or send out the wrong signal) it is more of a document ingestion module that support most of the prevelant file type and ouputs them in standerdised format ready to be used by downstream modules. So the thing starts with identifying the file types. Currently, it supports PDFs (.pdf), images (.png, .jpeg, .bit, .tiff and other prevelent types), sheets (.xls, .xlsx, .csv ), DOCs (.doc,.docx), PPTs (.ppt,.pptx), other straightforward types like .txt, .rtf, .md, .html, .xlm etc (these ones are added for sake for completion i.e they are less used in our production worflow that is supposed to be built for AP  and AR flow as these are more informal kinds of formats, and not generally used in finance accounting workflows, So maybe we should not highlight them much. Like they are there, but not to be stressed.). These could further be divided into major four categories. 
1. PDFs and Images
2. Sheet-like documents like Excel and CSV
3. More structured formats like Docs and PPTs
4. The fourth are like the unimportant ones, like that can be read straight forwardly like the text files, Rich Text Format, Markdown. 
For the first type that is a PDF and image, Which is the major focus and where the most effort has been. For these page-wise images are extracted. For images, it would be a single page, if it is a multi-page TIFF page-wise images are again extracted like pdfs. Then, we'll do page processing. Each page goes through a modular pipeline of:
1. Page angle classification
2. Text detection (which also incorporates our own algorithm of skew correction).
3. Direction classification (which is essentially says if a given detected bounding box is upright 0 degrees or upside down 180 degrees). The skew correction overlaps with the text detection and the direction classification modules, It uses the information from both of these modules to correct skew. Essentially, these two modules are anyway common OCR pipeline. So we utilize their output to perform skew correction by designing on an algorithm around it. Why we needed it? One existing classifier modules (0,90,180,270) sometime gave incorrect results and are not entirely perfect (also dont correct if skew is present). Standalone skew correction algorithms added computational overhead whilst not being completely robust. So we built detected bounding box and bound box direction classification existing outputs to provide more accurate and faster output for our use case We were already going to use these two modules, so they don't add any sustantial additional computation overhead "when not required" (I say this because, for most of the pages, they won't be rotated or queued, so the pipeline works normally, as it would, without much computation. It does not require any additional steps. For cases where the page classifying module may go wrong or there is some skew, there are additional detect (maybe one additional detection step) that happens. But it would be only for those cases, and while we don't add any additional competition overhead for most of the pages. Whereas, if we would have implemented a skew correction algorithm, it would have required us to do it for every page, and sometimes they are not very accurate and can be computationally costly.)
4. Once the bounding boxes are finalized through our detection and integrated skew-correction logic, we perform a final metadata injection step. Rather than sending raw crops directly to the recognition module, we attach "intelligence tags" to each box. This allows the recognition engine to adapt its behavior based on the confidence and geometry of the detection, providing a level of control that standard "off-the-shelf" OCR models cannot match.  
5. Once the recognition is done, we have the final OCR output - detection bounding box, and corresponding  - direction (0/180), alignment (vertical or horizontal) and text (the text in that bbox). This is then used as an input to the layout aware text conversion module that uses "layout as text" (using whitespace and newline) to get a spatial text representation of the page ready for llm or ther modules input down the line. The algorithm has different aspects to it. We have worked on physics-based (lack of better word, here i mean to say that the constrainst on how things are arranged in a documents like ex two bounding box that have significant x overlap wont belong to same line etc) constraints and dynamic spacing (as we have sort of single font size in raw string format while in document we encounter various font that we then have to represent in this sort of monofont format, which means the spaning has to be taken care of to get good spatial resembelence between the document and its spatial representation). These are not formal terms, but this does mean that this is not a simple algorithm, and we have integrated nuanced things to make the representation as good as possible. 
classification and skew correction, and finally, we'll do OCR. Spatial or layout-aware formatting is that OCR is converted to layout-aware formatting. Then, finally, we release our output: page-wise OCR, page-wise layout-aware text, page-wise images. .it will ouput pagewise layout aware text and pagewise processed image in numpy format - after being read and angle (0,90,180,270) and skew corrected, for excel it will output worksheet wise image and  
"""
- Designed modular OCR pipeline integrating PaddleOCR components (detector, recognizer, line classifier)
- Built comprehensive file reading module supporting PDF, images (JPG, PNG, TIFF, BMP), multipage TIFFs, Excel (XLS, XLSX, CSV), Word (DOC, DOCX), PowerPoint (PPT, PPTX), and plain text files
- Developed robust page orientation detection and skew correction module using text detection bounding boxes
- Designed spatial text (typewriter format) conversion algorithm preserving layout information for LLM ingestion
- Created Excel-to-spatial-text conversion handling merged cells, varying cell sizes, and text alignments


#### 3. DocSync Service
- Engineered document traceability and QA service across finance automation pipeline
- Integrated Human-in-the-Loop verification via Label Studio for daily review of misclassified documents
- Developed DocSyncRef pipeline using LayoutLM embeddings for layout-similarity filtering to create non-redundant reference datasets

#### 4. De-identification Framework (De-DoCS) [Ongoing]
- Designing framework for document de-identification enabling privacy-preserving data sharing
- Contextually consistent synthetic substitution of PII while preserving semantic fidelity
- Built annotation framework and 500-sample dataset; benchmarking OCR+LLM vs VLM approaches

#### 5. FinePDF Financial Document Corpus
- Filtered financial documents from FinePDFs corpus (3T tokens, 475M documents)
- Two-stage filtering: keyword-based + XGBoost classifier trained on LLM-labeled samples
- Resulted in ~120B token finance-focused dataset for continual pre-training

#### 6. Expense Management POC
- Designed end-to-end pipeline for automatic expense report generation from PDFs and images containing single or multiple receipts
- Pipeline: page/receipt segmentation, OCR, per-receipt LLM extraction, and final LLM-based expense report generation

---

## Publications

### Primary Author:
1. **LayoutWeaver: An Interactive System for Layout-Aware Text Correction** (Demo)
   - **WACV 2026** (Accepted)
   - Interactive human-centered system for correcting OCR-derived layout renderings
   - Achieved 10x higher annotation throughput compared to text-based correction

### Co-Author:
2. **SAVIOR: Sample-efficient Adaptation of Vision-Language Models for OCR Representation**
   - **VisionDocs Workshop, WACV 2026** (Accepted)
   - Sample-efficient data curation for robust financial OCR
   - Proposed PaIRS metric for layout fidelity evaluation
   - Fine-tuned Qwen2.5-VL achieving 0.93 word recall, 0.89 F1 on document QA

3. **DocuLite: A Scalable and Privacy-Preserving Framework for Financial Document Understanding**
   - **Deployable AI Workshop, AAAI 2026** (Accepted)
   - Synthetic data generation framework (InvoicePy + TemplatePy) for LLM/VLM training
   - Achieved 0.525 F1 improvement for LLM, 0.513 for VLM over UCSF baseline

---

## Academic Research (M.Tech)

### M.Tech Project: Multimodal Audio-Visual Affect Analysis
*Guide: Prof. Preeti Rao, IIT Bombay* | Jan 2023 - May 2024

- Developed ICA-UAVM (Interactive Cross-Attention Unified Audio-Visual Model) for audio-visual emotion recognition
- Explored multimodal fusion methods to combine audio and visual modalities effectively
- Evaluated on standard emotion benchmarks: RAVDESS, SAVEE, CREMA-D, MSP-IMPROV
- Extended to cross-cultural attitude analysis (16 classes) across German, Hindi, and Cantonese speakers using prosodic and visual cues

---

## Technical Skills

- **Languages:** Python
- **ML/DL:** PyTorch, Scikit-learn
- **LLMs/VLMs:** Prompt engineering, fine-tuning, multi-step pipelines
- **Tools:** LaTeX, Label Studio, Figma

---

## Scholastic Achievements
- JEE Advanced: 99.4 percentile among 0.2 million candidates (2018)
- JEE Mains: 99.8 percentile among 1 million candidates (2018)
- KVPY Fellowship by Department of Science and Technology, Government of India (2017)

---

## Areas of Expertise (for DeepMind application)
- Document AI & Understanding
- Vision-Language Models
- Multimodal Learning (Audio-Visual)
- OCR & Layout Analysis
- Information Extraction from Semi-structured Documents
