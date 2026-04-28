# Problem Description

## Original User Prompt

"Given that all the models we try only to detect the ball and tennis serves in my video, I plan to make my own fine tune of the yolo model to idenfy and track the tennis ball through the video. The video is basically me practicing tennis serves on my backyard(no tennis court). The ball traves at most 100km/h on the video and there are 8 tennis serves in the video. It is recorded on a iPhone 14 pro max camera in 4k and 60fps. The camera is positioned sideways from me and from below, tilted upwards, so it does not have the perfect angle, but you clearly can spot the ball in all serves. So to fine tune yolo, I have a few doubts first, like: 1. Is it possible for me with a local amd 6900xt amd gpu with 16gb dram fine tune this model? 2. I'm planning to fine tune the latest yolo model the yolo26 https://docs.ultralytics.com/models/yolo26/ is it possible to fine tune him? Or we must use an earlier version? 3. Can we fine tune using only our current video, that has more than 4500 frames? It will be enough to produce results and improve detection from videos recorded in the same way? 4. What would be the best way to produce the training dataset data? Do we need to manually train each frame? 5. Can we make an actual cli that automatically helps to categorize the dataset, what I want is, a cli that can automatically split the training frames, get the rjtpp yolov8 prediction for every training frame, and then show the user the image + where it was predicted. If is correct the user just click or go next, if not the user clicks where the ball is and then we update the training data. The cli also organizes the dataset correctly in the way we'll use it for training/fine tuning the new yolo model."

## User Clarifications

The user clarified the following after the initial problem-framing questions:

- Previous models do not detect the ball as well as expected.
- The best current model is RJTPP YOLOv8, but it still does not produce enough ball detections.
- The immediate scope should be ball detection only.
- Reliable detection means knowing where the tennis ball is in each image/frame.
- The target improvement is to detect better than the RJTPP YOLOv8 model on the user's video.
- The model should work best for the user's current setup, with possible generalization to similar setups.
- The AMD 6900XT has 16GB VRAM on the GPU itself.
- The RJTPP model is https://huggingface.co/RJTPP/tennis-ball-detection.
- The user accepts using whichever annotation format is needed.
- The user wants the assistant to decide later whether no-ball frames help or hurt training.
- The user can record additional videos if that improves fine-tuning quality.
- The user's local environment is NixOS with a Radeon 6900XT.
- The RJTPP YOLOv8 model definitely fails post-contact, and it also fails in other parts of the serve.
- The target metrics are precision and more detected frames.
- Similar setups mean videos recorded with a similar camera setup, similar distance from the ball, and a similar backyard/non-court environment.
- A browser-based annotation/review UI is preferred because it leaves room to improve the tool over time.

## Restated Problem

The user wants to understand whether it is feasible and worthwhile to fine-tune a YOLO model to detect and track a tennis ball in their own serve-practice video, because existing models have not met their needs. They also want to understand what data is needed, whether one 4K 60fps video with 8 serves is enough, whether their local AMD GPU can be used, whether YOLO26 is an appropriate fine-tuning target, and whether a CLI-assisted annotation workflow could reduce manual labeling by showing predictions for user confirmation or correction.

## Core Unknown

Main unknown: whether the user's available data, 16GB VRAM AMD 6900XT hardware, intended YOLO model, and desired annotation workflow are sufficient and appropriate for creating a tennis-ball detector that outperforms the RJTPP YOLOv8 model on the user's backyard serve videos.

Secondary unknowns:

- Whether the AMD 6900XT setup with 16GB GPU VRAM is capable of fine-tuning the chosen YOLO model in the user's local environment.
- Whether YOLO26 can be fine-tuned with the intended tooling, or whether an earlier YOLO version is required.
- Whether one current video with more than 4500 frames and 8 serves is enough training data for the intended detection/tracking quality and generalization scope.
- How much manual annotation is actually required, and whether every frame must be labeled.
- What exact annotation target is needed for the chosen training workflow: ball center point, bounding box, segmentation mask, or another label.
- What the proposed CLI should do, what interface it should expose, and what dataset format it should produce.
- What "rjtpp yolov8 prediction" refers to.

## Given Facts

- The user has already tried multiple models, and those models have not produced the desired tennis ball and serve detection results.
- The best current model tried by the user is the RJTPP YOLOv8 tennis-ball model, but it still misses more ball detections than expected.
- The user is considering fine-tuning a YOLO model.
- The immediate intended object of interest is the tennis ball, and the desired outcome is frame-by-frame ball detection.
- The source video shows the user practicing tennis serves in a backyard, not on a tennis court.
- The tennis ball travels at most about 100 km/h in the video.
- The video contains 8 tennis serves.
- The video was recorded on an iPhone 14 Pro Max camera.
- The video is 4K and 60fps.
- The camera is positioned sideways from the player, below the player, and tilted upward.
- The camera angle is not perfect.
- The user says the ball can clearly be spotted in all serves.
- The current video has more than 4500 frames.
- The user has a local AMD 6900XT AMD GPU and mentions "16gb dram".
- The user clarified that the AMD 6900XT has 16GB VRAM on the GPU itself.
- The user is on NixOS with a Radeon 6900XT.
- The user is considering YOLO26 and provided the Ultralytics YOLO26 documentation URL.
- The user wants to know whether YOLO26 can be fine-tuned or whether an earlier version is needed.
- The user wants to know whether a single current video is enough to improve detection for similarly recorded videos.
- The user can record additional videos if that would improve fine-tuning quality.
- The user wants to know the best way to produce training dataset data.
- The user wants to avoid manually labeling every frame if possible.
- The user accepts using whatever annotation representation is required by the training workflow.
- The user wants the assistant to decide later whether adding no-ball frames is beneficial or harmful for the training objective.
- The user identified post-contact ball detection as a definite RJTPP YOLOv8 failure area, with additional misses in other serve phases.
- The user wants to improve both precision and the number of frames where the ball is detected.
- The user prefers a browser-based review UI for the dataset annotation workflow.
- The user is interested in a CLI that splits training frames, runs an existing YOLOv8 prediction for each training frame, displays the image plus prediction, lets the user confirm correct predictions, lets the user click the correct ball location when predictions are wrong, updates the training data, and organizes the dataset for future fine-tuning.

## Conditions and Constraints

- The solution space is centered on YOLO fine-tuning, not on a generic computer-vision approach, although the exact YOLO version is still unresolved.
- The target video domain is narrow: backyard tennis serve videos recorded in a similar way.
- The success target is improvement over the RJTPP YOLOv8 tennis-ball detector for this video/setup.
- Improvement should be evaluated by precision and by increasing the number of frames with a detected ball.
- The target footage is high resolution and high frame rate: 4K at 60fps.
- The ball is small, fast-moving, and may be affected by the non-ideal camera angle.
- The dataset source currently described is a single video with 8 serves and more than 4500 frames, with the option to record more videos if useful.
- The user prefers a local workflow and specifically asks about local AMD GPU feasibility.
- The local workflow must account for NixOS and Radeon 6900XT/AMD GPU constraints.
- The desired dataset creation workflow should be assisted and interactive, not fully manual frame-by-frame labeling unless that is unavoidable.
- The proposed CLI must be able to show images and predictions, accept user confirmation/correction, and organize training data into the format needed for fine-tuning.
- The annotation/review interface should be browser-based rather than a pure terminal/OpenCV-window UI.
- The final model is expected to improve detection for videos recorded in the same way, not necessarily for arbitrary tennis videos.
- "Videos recorded in the same way" means a similar camera setup, similar distance from the ball, and a similar backyard/non-court environment.

## Missing Context

Required before solving:

- Whether the NixOS/Radeon 6900XT environment already supports AMD GPU acceleration for the intended training framework.
- Which exact YOLO26 weights/package/API the user intends to use and whether that tooling is already installed in the project environment.
- Whether annotations should ultimately be bounding boxes, center points converted to boxes, masks, or another format required by the selected training workflow.
- Whether frames without a visible ball should be included and how they should be labeled.
- Whether training and validation should be evaluated on the same video, held-out parts of the same video, or separate videos.
- How many additional videos are useful to record, and whether they should vary camera position, lighting, background, player clothing, or ball visibility while staying close to the target setup.
- Whether privacy/local-only processing is required because the footage is from the user's backyard.

Useful but not blocking:

- The exact video filename and current project command used to process it.
- Example frames where previous models fail and where the ball is visible.
- Whether the ball is yellow/green and how it contrasts with the background.
- Whether there is motion blur, occlusion by the body/racket, or compression artifacts.
- Whether the user is willing to label a smaller sampled subset manually if needed.
- Whether the CLI should be graphical via an OpenCV-style window, browser-based local UI, terminal-driven with external image display, or another interface.
- Whether the final dataset should live inside this repository or outside it.

## Ambiguities

- "Fine tune YOLO to identify and track" originally mixed detection and tracking. The user clarified that the immediate scope is ball detection only, but future tracking/serve analysis may depend on reliable detection.
- "Tennis serves" could mean recognizing serve attempts as events, detecting the player motion, detecting the racket/ball contact moment, or only using serve count as context.
- "YOLO26" needs validation as a specific model/version/toolchain target before any plan can assume compatibility.
- "Precision and more detected frames" define the target direction, but the exact validation protocol still needs to be chosen during planning.
- "Current video has more than 4500 frames" does not clarify how many frames actually contain a visible ball or a useful training signal.
- "Clearly can spot the ball" is a human visibility claim; it does not define annotation precision, bounding box size, blur level, or detector difficulty.
- "Similar setup" now means similar camera setup, similar distance from the ball, and similar backyard/non-court environment, but the acceptable variation range still needs to be defined during dataset design.
- "Automatically helps to categorize the dataset" could mean annotation, dataset splitting, quality review, class assignment, or all of these.
- "If correct the user just click or go next" needs UX definition: keyboard shortcuts, mouse clicks, auto-advance, correction mode, undo, zoom, and review behavior.
- "Click where the ball is" could produce a point annotation, but the required YOLO dataset format may require a bounding box or another representation.
- Whether no-ball frames should be used is intentionally delegated to the assistant for later planning, but the training objective and false-positive tolerance still need to be defined first.

## Risky Assumptions

- Assuming YOLO fine-tuning alone will eventually solve tracking could be wrong if later tracking failures are caused by temporal continuity rather than per-frame detection.
- Assuming one 4500-frame video is enough could be wrong because adjacent video frames may be highly correlated and may not represent varied backgrounds, lighting, ball sizes, or serve phases.
- Assuming the AMD 6900XT can train the selected model without checking the user's software stack could lead to a plan that fails locally.
- Assuming YOLO26 is fine-tunable through the same workflow as earlier Ultralytics models could be wrong if APIs, weights, licensing, or hardware support differ.
- Assuming every frame needs manual labeling could make the workload seem larger than necessary, but assuming few labels are enough could produce an unreliable dataset.
- Assuming the annotation should be a clicked point could be wrong if the training format requires boxes or if point-to-box conversion is insufficient for small, blurred balls.
- Assuming future videos are similar enough to the current video could overstate how well a model trained on one clip will generalize.
- Assuming RJTPP YOLOv8 predictions are good enough for assisted labeling could fail if its misses are concentrated in the most important frames.
- Assuming no-ball frames are irrelevant could create a detector with poor false-positive behavior.
- Assuming the desired CLI is purely terminal-based could conflict with the need to display high-resolution images and collect clicks accurately.

## Clarifying Questions

No blocking clarification questions remain for the understand step. The next step can decide implementation details such as the exact validation protocol, no-ball-frame policy, annotation representation, training stack, and browser UI shape.

Useful questions for planning:

1. Should the first training/evaluation pass use only the existing video, or should the user record additional videos before any training begins?
2. Should the browser review tool live inside the existing `web/` app or be a separate local annotation utility?
3. Should the first evaluation compare against RJTPP YOLOv8 on every sampled frame or on a manually reviewed validation subset?

## Evidence Log

- Existing model dissatisfaction: "all the models we try only to detect the ball and tennis serves in my video"
- Clarified model failure: "they dont detect the ball as good as I'm expecting"
- Current best baseline: "the best one is the rjtpp yolov8 model"
- Remaining baseline limitation: "still dont produce enough ball detections as expected"
- Intended approach: "I plan to make my own fine tune of the yolo model"
- Target object and behavior: "to idenfy and track the tennis ball through the video"
- Clarified immediate scope: "for now consider ball detection only"
- Clarified frame-level requirement: "frame by frame we must know where the tennis ball is on the image"
- Clarified target improvement: "detecting best than the rjtpp yolov8 model is where I'm aiming for my video"
- Clarified generalization scope: "it should work best for this setup, but we can generalize it with setups that's close as this"
- Recording scenario: "me practicing tennis serves on my backyard(no tennis court)"
- Ball speed: "The ball traves at most 100km/h"
- Serve count: "there are 8 tennis serves in the video"
- Recording device and format: "recorded on a iPhone 14 pro max camera in 4k and 60fps"
- Camera position: "positioned sideways from me and from below, tilted upwards"
- Camera limitation: "it does not have the perfect angle"
- Visibility: "you clearly can spot the ball in all serves"
- Hardware: "local amd 6900xt amd gpu with 16gb dram"
- Clarified hardware memory: "vram on the gpu itself"
- Clarified OS/GPU environment: "nixos and radeon 6900xt"
- Clarified failure phase: "post contact definitelly, but even other parts fail"
- Clarified metric direction: "precision and more detected frames"
- Clarified similar setup boundary: "similar camera with similar setup like the distance I'm from the ball and the kind of overall environment, not in a tennis court"
- Clarified UI preference: "browser is better cos it gives the ability for us to improve it"
- Candidate model: "the latest yolo model the yolo26 https://docs.ultralytics.com/models/yolo26/"
- Baseline model URL: "https://huggingface.co/RJTPP/tennis-ball-detection"
- Dataset size: "only our current video, that has more than 4500 frames"
- Generalization target: "videos recorded in the same way"
- Dataset-production concern: "What would be the best way to produce the training dataset data?"
- Manual-labeling concern: "Do we need to manually train each frame?"
- Annotation format delegation: "yes"
- No-ball frame decision delegation: "you decide, is it good for training adding no ball frames?"
- Additional video availability: "yes, I can record if it would the fine tuning better and improve it"
- Desired annotation assistant: "a cli that can automatically split the training frames, get the rjtpp yolov8 prediction for every training frame"
- Review UX: "show the user the image + where it was predicted"
- Confirmation/correction UX: "If is correct the user just click or go next, if not the user clicks where the ball is"
- Dataset organization: "The cli also organizes the dataset correctly in the way we'll use it for training/fine tuning the new yolo model"

## Readiness Assessment

Ready for planning.

The core task is clear enough to proceed to George Polya's next step: make a plan. The immediate target is frame-by-frame tennis-ball detection that outperforms the RJTPP YOLOv8 baseline on the user's NixOS/Radeon 6900XT backyard serve setup, especially post-contact. Remaining details such as the no-ball-frame policy, exact validation split, annotation representation, YOLO version choice, AMD training setup, and browser UI shape are planning decisions rather than blockers to understanding the problem.
