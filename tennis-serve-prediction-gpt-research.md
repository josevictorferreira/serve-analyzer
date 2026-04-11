# Viabilidade de um app que mede velocidade e “ponto de queda” do saque usando vídeo no quintal

## É possível, mas não do jeito “totalmente automático” sem algum tipo de calibração

Um app que **detecta a bola**, **reconhece o gesto do saque**, **identifica o instante do impacto raquete–bola** e **estima velocidade/direção após o contato** é tecnicamente viável com visão computacional moderna — há literatura robusta para rastreamento de bolas pequenas e rápidas, e pipelines de detecção + filtragem (ex.: redes + Kalman) são comuns em esportes. citeturn9view3turn9view1turn17view1turn10view3

O que muda o jogo é o seu requisito mais “físico”: **converter pixels em metros/segundo e prever onde a bola cairia numa quadra real** a partir de uma câmera em ângulo livre, no quintal, sem linhas de quadra. Essa parte **pode ser feita**, mas, para ser **confiável**, quase sempre exige pelo menos **uma fonte de escala e referência geométrica** (mesmo que mínima), porque:
- velocidade em m/s depende de **escala métrica**;
- direção 3D (incluindo componente vertical) e “ponto de queda” dependem de **trajetória 3D**;
- com **uma única câmera RGB** e sem referências no ambiente, você tende a ter **ambiguidade e erro** no “mundo real”, especialmente num objeto **pequeno, rápido e com blur** como a bola de tênis. citeturn9view3turn9view1turn20view0turn12view0

Um indício bem prático de por que apps “de saque” geralmente pedem **quadra visível**: há app comercial que afirma calcular velocidade do saque porque **reconhece as linhas da quadra** no frame e, com isso, diz reconstruir coordenadas 3D do jogador e da bola a partir de coordenadas na imagem (i.e., usa a quadra como calibrador). citeturn5view0  
Em outras palavras: **não é coincidência** que a “quadra” apareça como parte do algoritmo em produtos que prometem velocidade. citeturn5view0

Conclusão direta (com o mesmo nível de exigência que você descreveu):  
- **Sim**, é possível construir o sistema. citeturn9view3turn17view1turn8view0turn10view0  
- **Mas** para entregar **velocidade em m/s + direção + ponto de queda numa quadra real** com **pouquíssima entrada do usuário** e **câmera em ângulo livre**, você quase certamente vai querer **(a)** algum tipo de **calibração leve** (ex.: marcador impresso no chão) **ou (b)** usar **sensores/AR** do celular para extrair pose/escala **ou (c)** **duas câmeras** (estéreo) para 3D mais estável. citeturn5view3turn11view0turn5view2turn14view0

## Como eu montaria o pipeline, passo a passo

A forma mais “engenheirável” de pensar nisso é decompor em 5 problemas acoplados: (1) detecção/track da bola, (2) detecção do corpo/raquete e reconhecimento do saque, (3) detecção do impacto, (4) reconstrução 3D/escala, (5) física para prever queda (e opcionalmente quique/spin). citeturn9view3turn8view0turn10view0turn14view0

**Passo a passo de um pipeline realista:**

Primeiro, você roda **pose estimation do corpo** para localizar o atleta, padronizar a ROI (região) e segmentar o tempo “onde há saque”. Soluções modernas como MediaPipe fazem rastreamento de pose com dezenas de landmarks e rodam em tempo real em mobile, e podem produzir landmarks 3D/“world coordinates” (dependendo do modo/modelo). citeturn8view0turn8view1turn8view3

Segundo, em paralelo ou logo depois, você faz **detecção/track da bola**. Para bola de tênis, o desafio é justamente ser **pequena e rápida**, frequentemente com blur, às vezes invisível/ocluída; por isso, arquiteturas específicas como TrackNet (heatmap + múltiplos frames consecutivos) foram propostas exatamente para “bola minúscula e rápida em esporte”, aprendendo padrão temporal além do frame isolado. citeturn9view1turn9view2turn9view3

Terceiro, você determina **o instante do impacto**. Existem várias estratégias:
- por visão clássica: detectar a **mudança brusca** na velocidade/direção estimada da bola e/ou a menor distância bola–raquete;
- por multimodalidade: usar áudio (microfone) como “pico” de contato e refinar por visão (isso é uma técnica prática, embora a robustez varie e depende do ruído ambiente);  
- por sensores especializados: há trabalho acadêmico usando **event cameras** para localizar impacto em tempo real, justamente porque conseguem registrar mudanças com precisão temporal muito alta. Esse trabalho também mostra um aspecto importante para você: **luz solar direta** pode derrubar taxa de detecção de contornos (um exemplo de “condição do mundo real” que mata a robustez). citeturn15view0

Quarto, com o impacto detectado, você estima o vetor velocidade **logo após o contato**. Em geral, você suaviza a trajetória com um filtro (Kalman é o clássico) e deriva velocidade do estado filtrado. OpenCV tem implementação direta de KalmanFilter, e há exemplos recentes em esportes usando “detector (YOLO) + Kalman + escala espaço-temporal” para estimar velocidade a partir de vídeo comum. citeturn10view3turn17view1turn9view3

Quinto, para prever “onde cairia numa quadra real”, você precisa de um **modelo de voo**. Um modelo puramente balístico (gravidade apenas) já dá algo, mas em bolas com rotação, o **efeito Magnus** pode desviar a trajetória e aparece explicitamente como fonte relevante de erro em estudos de previsão de ponto de queda; em tênis, trabalhos de múltiplas câmeras reforçam que velocidade e spin são estados ocultos relevantes, e que incorporar spin melhora previsão de landing. citeturn14view0turn13view1turn13view3

## A parte crítica: como obter posição/ângulo da câmera e escala sem “inputs chatos”

Há dois “tipos de calibração” que você precisa separar: **intrínsecos** (lente/câmera — focal, distorção) e **extrínsecos** (pose da câmera no mundo — posição e orientação). Em visão computacional clássica, estimar pose a partir de correspondências 3D↔2D é o problema PnP (solvePnP no OpenCV). citeturn10view0turn10view1

A questão é: **de onde vêm os pontos 3D conhecidos** se você não tem quadra e não quer pedir input?

A opção mais robusta e com melhor custo-benefício costuma ser **um marcador fiducial impresso** (ou “board” de marcadores) no chão/parede, que serve como régua + referência de orientação. O próprio tutorial do OpenCV para ArUco é explícito: para estimar pose você precisa de parâmetros de calibração (matriz da câmera e distorção) e do **tamanho real do marcador**; e boards aumentam robustez a oclusão justamente por usar vários cantos/pontos. citeturn5view3turn11view2turn10view0  
De modo similar, AprilTag fornece rotina de pose que usa **tag size** e intrínsecos (fx, fy, cx, cy) e até sugere usar solvePnP (incluindo variante adequada para quadrados) como alternativa. citeturn11view0turn10view1

Uma segunda opção, mais “mágica” para o usuário (menos papel no chão), é usar AR do celular para obter a pose da câmera via VIO/SLAM. A documentação do ARCore descreve que ele combina **informação visual + IMU** para estimar a pose da câmera ao longo do tempo e também detecta planos e pode fornecer mapas de profundidade (dependendo do device). Isso te dá um referencial “mundo AR” que pode ajudar a reduzir inputs. citeturn5view2turn6view3  
No ecossistema da entity["company","Apple","consumer electronics company"], a visão oficial de ARKit destaca recursos como detecção de planos/scene understanding em devices com LiDAR e “Motion Capture” com uma câmera, com menção a melhorias de estimativa de altura em gerações específicas — o tipo de sinal que pode ajudar a ancorar escala/pose do corpo e, indiretamente, posicionar a câmera “em relação a você”. citeturn7view0

A terceira opção é “monocular puro” usando **dimensão conhecida do objeto** (bola) para inferir profundidade/escala. A bola de tênis tem faixa de diâmetro regulamentar; a entity["organization","International Tennis Federation","tennis governing body"] publica especificações (ex.: tamanho 6,54–6,86 cm para tipos comuns, além de massa etc.). citeturn12view0  
Dá para explorar “cor + tamanho” com uma câmera só: há tese acadêmica descrevendo rastrear bola em 3D a partir de vídeo monocular justamente explorando cor e dimensões conhecidas, e ainda enfatiza a caracterização de incerteza por ser reconstrução monocular. citeturn20view0  
O problema prático é que, em saque, a bola frequentemente tem blur e muda de escala muito rápido; então medir “diâmetro aparente” com precisão o suficiente para metrificar velocidade pode ficar instável em condições reais. citeturn9view3turn20view0

image_group{"layout":"carousel","aspect_ratio":"1:1","query":["OpenCV ArUco marker board printed on paper","AprilTag fiducial marker printed","smartphone tripod tennis serve side view backyard","tennis ball tracking heatmap TrackNet visualization"],"num_per_query":1}

## Como calcular velocidade/direção e projetar ponto de queda numa quadra real

### Velocidade e direção imediatamente após o impacto

Em termos matemáticos, você quer o vetor velocidade 3D **v₀** logo após o contato. O que você mede do vídeo é uma sequência de observações 2D (u, v) do centro da bola por frame (e, às vezes, tamanho aparente/blur). Para sair de 2D→3D, você precisa de:
- intrínsecos (K) e distorção;
- uma forma de obter profundidade Z ao longo do tempo (ou triangulação por múltiplas vistas);
- filtragem/suavização para derivar velocidade sem amplificar ruído. citeturn10view0turn10view1turn10view3turn9view3

Na prática, há três caminhos:

**Caminho A: estéreo / múltiplas câmeras (o mais confiável para 3D)**  
Pesquisa em robótica e visão para esportes frequentemente usa estéreo/múltiplas câmeras para localizar bola em 3D e prever trajetória. Um paper recente para tênis enfatiza explicitamente “multi-camera” e estima estados ocultos (velocidade, spin) para melhorar previsão de landing. citeturn14view0  
E mesmo em literatura de produto/sistema, é comum reconhecer que a forma tradicional “cara” para velocidade de bola envolve múltiplas câmeras high-speed com visão computacional. citeturn16view1

**Caminho B: monocular + referência geométrica fixa (quadra, marcador, plano)**  
Se você tem uma referência fixa no mundo (linhas de quadra, um board ArUco, etc.), você consegue ancorar pose e escala com solvePnP e daí converter trajetória. O OpenCV documenta solvePnP como solução para obter rotação/translação de pose a partir de pontos 3D↔2D, e tutoriais ArUco explicam como isso vira “pose da câmera em relação ao marcador”. citeturn10view1turn5view3turn11view2  
Isso se aproxima do que apps comerciais fazem ao “reconhecer quadra” para reconstruir coordenadas 3D da bola/jogador. citeturn5view0

**Caminho C: monocular “puro” + tamanho conhecido + modelos probabilísticos (o mais frágil)**  
Funciona em cenários controlados e existe trabalho acadêmico com reconstrução 3D a partir de uma câmera usando cor+dimensão da bola e Kalman, mas a própria tese aponta a necessidade de modelar incerteza e mostra como erro se correlaciona com profundidade e condições de segmentação. citeturn20view0turn10view3

### “Ponto de queda” e mapeamento para uma quadra real

Para prever onde a bola cai, o modelo mínimo é: posição inicial + velocidade inicial + gravidade → interseção com o plano do chão. Mas em esportes de bola, rotação pode desviar. Um estudo em tênis que avalia previsão de landing em diferentes campos reporta que a acurácia (especialmente em um eixo) é sensível a fatores de perturbação e identifica explicitamente o **efeito Magnus** como contribuinte importante. citeturn13view1  
Outro trabalho (robótica) deixa claro que spin “complica” a previsão e que incorporar spin como estado oculto melhora muito o erro de landing (relatando redução percentual grande vs. baseline). citeturn14view0  
E um artigo técnico de robô de tênis de mesa reforça a mesma ideia física geral: spin tem grande impacto na trajetória; inclusive menciona limitações de frame rate para conseguir estimar rotação com câmera padrão. citeturn13view3

Agora o detalhe que conecta com seu quintal: “ponto de queda numa quadra real” exige um **sistema de coordenadas de quadra**. Se você não tem quadra no vídeo, então você precisa definir um “mapeamento virtual”, por exemplo:
- origem na posição dos seus pés no impacto;
- direção “para frente” inferida pela orientação do tronco/ombros (pose);
- escala definida por marcador/AR/altura;  
e então projetar o ponto de interseção num **template padrão** (dimensões oficiais). Sem isso, o app até pode dizer “vai cair X metros à frente e Y metros à direita”, mas não consegue afirmar “cai no T” com confiabilidade. citeturn8view0turn8view1turn10view1turn5view0

## Bibliotecas, modelos e “blocos” que valem seu tempo

Abaixo está uma lista “orientada a implementação” (mobile e/ou backend), mas focada no que você pediu: **cálculo e visão**.

Para detecção e rastreamento da bola, você vai alternar entre (a) detectores genéricos otimizados e (b) modelos especializados em “bola pequena e rápida”. TrackNet é um exemplo clássico de especialização: ele usa frames consecutivos e produz heatmap para posicionar bola mesmo com blur/oclusão (há implementações não-oficiais em PyTorch com dataset descrito). citeturn9view1turn9view2  
Para detectores genéricos, a entity["company","Ultralytics","yolo developer company"] mantém documentação de YOLOv8, com foco em trade-off velocidade/precisão e uso amplo em detecção em tempo real. citeturn1search3turn1search30  
Um ponto realista: revisões recentes sobre detecção de bolas enfatizam desafios como oclusão, fundo dinâmico, variação de iluminação e alta velocidade — e que “detectar objetos minúsculos” é um gargalo recorrente. citeturn9view3

Para reconhecer o gesto do saque e suas fases, o caminho mais “industrial” é reduzir vídeo a um conjunto estável de features (esqueleto/landmarks) e fazer classificação temporal. MediaPipe Pose fornece 33 landmarks e a própria documentação posiciona como “alta fidelidade” e performance em dispositivos comuns. citeturn8view0  
Se você quer uma API pronta (Android/iOS), o ML Kit Pose Detection descreve 33 pontos e fornece scores por landmark, além de opções “base vs accurate” com diferentes taxas de frame. citeturn8view3

Para “entender o vídeo” diretamente (sem passar só por skeleton), arquiteturas de reconhecimento de ação são referência:  
- SlowFast (ICCV 2019) usa dois caminhos — um lento para semântica espacial e um rápido para movimento — explicitamente para capturar dinâmica temporal fina. citeturn3search26  
- I3D (Inflated 3D ConvNet) é um marco em ação em vídeo (3D conv inflada) e tem modelos disponibilizados por autores em repositório. citeturn3search3turn3search11  
Esses modelos não são “de tênis” por padrão, mas você pode fine-tunar para “saque / não-saque” e até fases do saque se tiver dataset anotado.

Para calibração, pose da câmera e escala, a combinação OpenCV calib3d + fiduciais costuma ser o “martelo certo”:  
- solvePnP e calib3d no OpenCV são os tijolos para pose (rotação/translação) a partir de correspondências 3D↔2D. citeturn10view1turn10view0  
- ArUco/boards no OpenCV têm tutoriais que explicam que pose precisa de intrínsecos e do tamanho do marcador, e boards aumentam robustez sob oclusão. citeturn5view3turn11view2  
- AprilTag tem documentação de pose com tag size e intrínsecos e menciona solvePnP como alternativa. citeturn11view0

Para filtragem e rastreamento (trajetória e derivadas), Kalman continua sendo “padrão ouro” para suavizar ruído antes de derivar velocidade. OpenCV documenta a classe KalmanFilter e papers recentes de esportes em mobile descrevem exatamente o padrão “YOLO + Kalman + estimativa cinemática com escala”. citeturn10view3turn17view1

Por fim, vale enxergar que soluções comerciais existem (com diferentes promessas). Um review de entity["organization","SwingVision","tennis analytics app"] descreve medição de velocidade e tracking como parte do pacote, mas como não é paper nem documentação técnica, eu trataria como “evidência de mercado”, não como prova de acurácia em condições fora do padrão (ex.: quintal sem quadra). citeturn19view0turn0search1

## Plano de execução prático com o mínimo de fricção e o máximo de chance de funcionar

O jeito mais rápido de sair do “paper design” e validar viabilidade é construir um MVP em camadas, onde cada camada responde uma pergunta objetiva.

Comece assumindo câmera fixa em tripé e capture com o máximo de qualidade possível (alto fps ajuda). A literatura reforça que bola pequena/rápida fica mais difícil com baixa taxa de quadros e baixa resolução, e isso aparece desde TrackNet até revisões amplas de detecção de bolas. citeturn9view1turn9view3

Em seguida:

Você implementa **ball tracking robusto** e mede velocidade em **pixels/segundo** (ainda sem converter para m/s). Isso já te permite validar se o rastreamento sobrevive ao blur e ao fundo do quintal. Use TrackNet/YOLO como baseline e um filtro (Kalman) para suavização. citeturn9view2turn17view1turn10view3

Depois, implemente **detecção do evento “impacto”** como um problema separado: encontre o frame em que a bola “se separa” da raquete e a velocidade muda. Compare com a abordagem de literatura (mesmo que com sensor especializado): o paper de event camera mostra decomposição do problema em “janela do swing → timing do impacto → contorno”, além de alertar sobre fragilidade sob sol direto. Isso te dá heurísticas de robustez (ex.: lidar com brilho/oclusão). citeturn15view0

Só então você entra no “hard mode”: converter pixel→m/s. Minha recomendação prática para preservar seu requisito de poucos inputs é:

- “Plano A” (melhor equilíbrio): um **board ArUco/AprilTag impresso** que o usuário coloca no chão a 1–2 m do jogador antes de gravar. É um input mínimo (colocar uma folha), mas resolve escala e orientação de forma muito mais estável do que tentar adivinhar tudo pela bola/corpo. citeturn5view3turn11view0turn11view2  
- “Plano B” (mais ‘mágico’, mais variável): usar ARCore/SLAM para obter pose e plano do chão, sabendo que a acurácia varia por device e sensores; e complementar escala por altura do corpo/estimativa do próprio AR quando disponível. citeturn5view2turn6view3turn7view0  
- “Plano C” (sem marcadores): usar tamanho regulamentar da bola + intrínsecos para inferir profundidade e aceitar erros maiores; a tese de reconstrução monocular ajuda a estruturar isso com incerteza, mas isso tende a ser sensível à qualidade de segmentação e blur (comum no saque). citeturn12view0turn20view0turn9view3

Para landing prediction, o MVP deve começar com modelo balístico e, só se você realmente precisar de precisão “quadra real”, adicionar termos de arrasto/spin. A importância do Magnus/spin em erro de landing aparece tanto em estudo específico de previsão de ponto de queda em tênis quanto em trabalho multi-câmera focado em velocidade+spin para melhorar landing. citeturn13view1turn14view0

Como alternativa que muita gente ignora: se seu objetivo principal é “feedback de evolução do saque”, pode ser estrategicamente melhor medir **velocidade relativa** e consistência (mesma câmera, mesma distância), e só ativar o modo “métrica absoluta e landing” quando o usuário aceitar o marcador/AR. Isso reduz fricção e aumenta taxa de sucesso do produto — e respeita o fato de que, para velocidade “real”, sistemas tradicionais recorrem a setups mais caros (múltiplas câmeras high-speed). citeturn16view1turn5view0