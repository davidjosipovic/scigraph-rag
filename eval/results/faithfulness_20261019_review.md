# Faithfulness sudac: neslaganja za pregled

Uzorak: 100 tvrdnji (`faithfulness_benchmark_20261019_*_sample.json`, 50 lokalnih + 50 Groq).
Claude je označio svih 100 naslijepo (`claude_label`), bez gledanja oznake suca (qwen3:8b).

- Slaganje: 79/100, Cohenov κ = 0.51 (S/U/C)
- Binarno (podržano / nije podržano): 84/100, κ = 0.60

**Pregled (2026-10-19): autor se složio sa svih 21 Claudeovih oznaka; `human_label` = `claude_label`.**

Ispod je 21 neslaganje. Za svako napiši konačnu oznaku: **S** (piše u kontekstu),
**U** (nema u kontekstu) ili **C** (kontekst kaže drugo). Ako se slažeš s Claudeom, ostavi.
Preostalih 79 slaganja ulazi u `human_label` kao Claude + sudac.

Kratice: *sudac* = konačna oznaka nakon provjere citata; *sirovo* = što je sudac sam rekao.

## Lokalni modeli (105454)

| # | Tvrdnja | Sudac | Claude | Zašto Claude | Konačno |
|---|---|---|---|---|---|
| 105454#4 | The papers do not explicitly mention the use of tabular data in the context of deep learning. | U | **S** | Tvrdnja o odsutnosti; "tabular" se u tom kontekstu stvarno ne pojavljuje. | |
| 105454#5 | Comprehensive Android Malware Detection … is particularly relevant in the context of mobile devices where privacy and computational efficiency are important considerations. | S | **U** | Kontekst ima samo datasetove i F1 = 98.53 %; privatnost i efikasnost na mobitelima su dodani. | |
| 105454#6 | The paper 'BoolQ …' (2019) used GPT-3.5-turbo. | U | **C** | BoolQ zapis navodi GPT-1, BERT i BiDAF; GPT-3.5 pripada drugom radu (CP-KGC). | |
| 105454#7 | Enhancing text-based knowledge graph completion … utilizes Android malware & network intrusion case studies. | U | **C** | To je dataset rada CADE, ne ovog rada. | |
| 105454#14 | The paper titled "IWSLT2015 German-English" uses the Penn Treebank (Word Level) dataset for machine translation. | U | **C** | IWSLT2015 je benchmark, a ne rad; PTB se u kontekstu koristi za language modelling, ne za MT. | |
| 105454#16 | The Imagenet-1K variant of the ImageNet dataset is used for evaluating the robustness of vision models. | S | **U** | InternVL za robusnost spominje varijante (A/R/Sketch); ImageNet-1K je osnovni skup, a "robusnost" za njega nije navedena. | |
| 105454#17 | The paper 'AnoShift …' studies contrastive learning for visual representations. | S | **U** | AnoShift je benchmark za detekciju anomalija; vizualne reprezentacije se ne spominju. | |
| 105454#19 | The paper "ST-MoE-BERT …" uses BERT as its model. | U | **S** | Piše `model: St-moe-bert`, model je izveden iz BERT-a. | |
| 105454#26 | The paper "Effective Handwritten Digit Recognition with Differentiable Spike …" mentions Mini-imagenet. | U | **C** | Naslov je izmišljen, a Mini-ImageNet je u drugom radu (Geometric Mean Loss); Dspike ga nema. | |
| 105454#32 | Decision fusion is a technique used in machine learning. | U | **S** | Kontekst ima `model: Decision fusion with gru`, pa je to izravna posljedica. | |
| 105454#35 | The paper 'KNAS: Green Neural Architecture Search' introduces a method for green (energy-efficient) neural architecture search. | U | **S** | Tvrdnja preformulira naslov. | |
| 105454#43 | The dataset used in the paper 'Improved Techniques for Training GANs' is Replica. | U | **C** | Improved GAN koristi CIFAR-10 i SVHN; Replica je dataset rada GAUDI. | |

## Groq modeli (121003)

| # | Tvrdnja | Sudac | Claude | Zašto Claude | Konačno |
|---|---|---|---|---|---|
| 121003#1 | The paper *Collection and Validation of Psychophysiological Data …* compares SVM and Random Forest for classification. | U (sirovo S) | **S** | Rezultati za RF i SVM su u kontekstu. Sudac je rekao S, ali je citirao `Random forest \| SVM`, retka kojeg nema, pa je provjera citata oznaku spustila na U. | |
| 121003#16 | The comparison is limited to the specific experiments reported in the cited papers. | U (sirovo S) | **S** | Meta-tvrdnja koja ništa ne izmišlja. Sudac je rekao S, ali bez citata, pa je oznaka postala U. | |
| 121003#18 | *Classifying Swahili Smishing Attacks …* reports 94.2 % accuracy, 93 % precision, 95 % recall, but does not identify whether the best model was RF or SVM. | U | **S** | Brojke se točno poklapaju, a koji je model najbolji zaista ne piše. | |
| 121003#22 | BERT uses Next Sentence Prediction (NSP) to train the model to determine whether two sentences appear consecutively in the original text. | S | **U** | NSP je u kontekstu, ali objašnjenje što NSP radi je pozadinsko znanje. | |
| 121003#24 | Paper "Acoustic measurements, field study, observational analysis, …" uses observational analysis | S | **U** | Navedeni "naslov" je zapravo popis metodologija; rad pod tim imenom ne postoji. | |
| 121003#26 | 'Deep Residual Output Layers …' evaluates both LSTM and Transformer architectures on WMT2014 English-German. | S | **C** | Na WMT2014 je samo Transformer-drill (BLEU 28.1); LSTM modeli su na PTB i WikiText-2. | |
| 121003#30 | HS3-Bench reports that DeepLab achieves an accuracy of 86.72% on the Hyko2-vis dataset. | S | **C** | Rezultati stoje iznad svog `model:` retka, pa je 86.72 rezultat RU-Neta, a DeepLabv3 ima 84.1. Isti raspored daje i GRU 60.6 kod HiRID-a. | |
| 121003#32 | BanglaCoNER fine-tunes Ft-bangla bert large and combines it with a POS-tagger, prefix-suffix features, k-neighbor words, and k-means clustering. | S | **U** | U kontekstu su to dva odvojena modela; da su kombinirani nigdje ne piše. | |
| 121003#39 | The paper *BoolQ …* is about natural language understanding tasks. | S | **U** | Kontekst kaže samo "Question answering"; NLU je dodan. | |

## Obrasci

- **Pogrešno pripisivanje (Claude C, sudac U)**: 5 slučajeva. Tvrdnja uzima podatak iz jednog rada i pripisuje ga drugom. Sudac to vidi kao "nema dokaza", a ne kao proturječje. Za Faithful (udio S) ova razlika nije bitna, ali udio C je zato podcijenjen.
- **Provjera citata ruši točne tvrdnje**: 2 slučaja (121003#1, #16). Sudac je točno rekao S, ali je citat složio od više redaka.
- **Pozadinsko znanje ili netočna tvrdnja (Claude U, sudac S)**: 7 slučajeva. Sudac propusti dodani detalj kad je glavni pojam u kontekstu.
- **Format redaka u ORKG kontekstu (121003#30)**: kod tablica s više modela sudac ne zna kojem modelu pripada rezultat.
