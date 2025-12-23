# Why this AI is needed

## differences of disorders of consciousness
- Brain death
- PVS / UWS -> 뇌는 살아있으나 의식은 없는 상태
- MCS -> 최소의식상태 (감정 반응 가능, 간헐적 명령 수행)
- Locked-in Syndrome (의식 100% 정상, 몸만 마비된 상태)

위에서부터 아래로 내려올 수록 뇌의 기능이 순차적으로 좋은 상태이다 (뇌사부터 locked-in syndrome 까지)

Brain Death는 PVS와 구분이 쉽다. 그러나 PVS와 MCS는 구분이 어렵다. (후술)

### The main point is PVS vs MCS diagnosis

현재 PVS/MCS 치료 목표:

PVS -> 지속적 식물상태
생명 유지 + 합병증 예방
의식 회복을 전제로 하지 않음, 즉 신경계 회복은 사실상 포기
-> 호흡 관리, 영양 공급, 욕창/감염 예방, 관절 구축 방지 (수동 재활)

사실상 환자 포기 상태임.

MCS -> 최소의식상태
의식 회복 + 기능 향상
회복 가능성 있음을 전제로 함, 즉 신경계 회복을 주된 치료 목표로 설정함
-> 집중적 신경재활, 능동적/자극 중심 치료, 비침습적 뇌자극 등등

### current diagnosis method for distinguish PVS and MCS
현재 CRS-R 평가로 PVS와 MCS를 구분하지만 CRS-R 평가의 경우 
Auditory function scale
Visual Function scale
Motor function scale
Oromotor/verbal function scale
Communication scale
Arousal scale
의 카테고리에서 평가자가 직접 평가하는 방식이며 (on paper) 또한 뇌의 각성 상태에 따라 다른 결과를 가지고 온다는 점, 실제로 여러번 진행해야 한다는 점, 30~40%의 확률로 MCS 환자가 PVS로 오진되는 등 문제점이 존재함.
https://www.jkna.org/journal/view.php?number=6676
https://pubmed.ncbi.nlm.nih.gov/25891806/

### How can we reduce misdiagnosis?
뇌파(electroencephalography, EEG)는 의식장애의 급성기에 뇌전증 활동(epileptic activity)을 확인하고 예후를 확인하는데 도움을 주지만 장기 의식장애에서는 비특이적인 전반적 서파가 주를 이루어 통상적으로 단시간 기록된 뇌파의 시각적 분석은 식물상태와 최소의식상태를 감별하거나 예후를 판단하기에 비교적 유용성이 낮다. 그러나 눈 감고 뜨기와 소리 자극, 광 자극 모두에서 뇌파 반응성을 보이는 경우에는 최소의식상태 진단에 높은 특이도와 낮은 민감도를 보이며, 세 가지 중 한 가지에서만 반응성을 보이는 경우는 높은 민감도를 보이지만 특이도가 낮다. 장기 뇌파(long-term EEG)를 이용한 연구에 따르면 식물상태에서는 수면과 각성의 행동 양상이 보인다 하더라도 뇌파에서는 수면-각성 패턴이 관찰되지 않는 반면 최소의식상태 환자에서는 행동상의 수면-각성 패턴에 따라 어느 정도의 뇌파 변화가 동반되며, 특히 행동상의 수면시에 램(rapid eye movement, REM) 수면과 비램수면 간의 전환 주기가 관찰된다

J Korean Neurol Assoc 2020; 38(1): 9-15.
Published online: February 1, 2020
DOI: https://doi.org/10.17340/jkna.2020.1.2

위에 설명되어 있는 대로 short-term EEG는 판단하기 적절하지 못하다, 또한 eye-blink, arousal, visual stimulation에 대한 EEG 반응도도 각각 위양성 위음성의 문제가 있다. 따라서 이미 연구에서 효과가 있다고 인정된 long-term EEG를 이용하여 더 높은 정확도를 가지는 뇌 신경 상태 측정 AI를 만들고자 한다.

### current problem with long-term EEG diagnosis and why are we using CRS-R?
현재 long-term EEG 대신 CRS-R이 널리 쓰이는 이유는 CRS가 훨씬 실용적이기 때문이다.

CRS-R은 추가 장비 없이도 반복 측정이 가능하고 또한 침상 바로 곁에서 시행할 수 있기 때문에 대부분의 환경에서 적용할 수 있으므로 큰 비용이 들지 않는다.

반면 long-term EEG는 장비, 인력, 시간이 많이 들어서 반복적으로 시행하기 어렵고 특히 장기 의식장애 환자 전체를 대상으로 screening, follow-up에 쓰기엔 현실적 제약이 크다.

또한 지금까지는 EEG를 즉석에서 잠깐 환자에게 연결해 의사가 live 하게 측정하는 법을 사용했기 때문에 그럴 바에는 CRS를 이용하는 것이 합리적이였기 때문에 CRS-R이 널리 사용된다.
https://www.jkna.org/journal/Table.php?xn=jkna-42-2-107.xml&id=

-> 그렇다면 비용 문제도 해결하고 전체 환자에게 적용할 수 있는, automatical monitoring이 가능한 Brainwave AI를 만들면 되는 것이 아닌가?
YES !!  

### how to make the AI?
- Machine Learning (Supervised)
Use PyTorch
MVP : CNN + LSTM
Better : CNN + Transformer (for foundation model)

As long as EEG is time series data, we need to use LSTM. But, according to the newly posted papers, we found transformer is better than LSTM to build universal (foundation) model. Therefore, for the MVP, I'm going to use CNN + LSTM + FC (fully connedted layers), and will reinforce it by exchange LSTM for Transformer.

선행된 연구 : https://pmc.ncbi.nlm.nih.gov/articles/PMC12416531/ (pubmed)

conclusion : 
In this study, EEG spectral analysis and complexity analysis were used as research methods to explore the EEG features potentially related to consciousness improvement in patients in PVS. The results showed that the SVM model featuring MSE demonstrated superior prediction performance in terms of classification accuracy and model evaluation, suggesting that MSE is a promising indicator for prognosis in patients in PVS. The underlying mechanisms of MSE’s predictive capability need to be further explored.

Lai K, Chen X, He L, et al. Identifying Features of Electroencephalography Associated with Improved Awareness in Persistent Vegetative State via Multiscale Entropy: A Machine Learning Modeling Study, Neurotrauma Reports 2025:6(1):720–731, doi: 10.1177/2689288X251369274.

First purpose : distinguish consciousness and unconsciousness brainwave frequency by FFT