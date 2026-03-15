<div align=center>
  <h1>
    Diffusion Synchronization
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs492d-fall-2024/ target="_blank"><b>KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)</b></a><br>
    Programming Assignment 5
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://jh27kim.github.io/ target="_blank"><b>Jaihoon Kim</b></a>  (jh27kim [at] kaist.ac.kr)
  </p>
</div>

<div align=center>
  <img src="./asset/teaser.png" width="768"/>
</div>


## Description
Image diffusion models trained on Internet-scale datasets can generate photorealistic images. However, they are limited in generating other types of visual content, such as ambiguous images and arbitrary-sized images that have different representations or resolutions. Diffusion synchronization aims to expand the capabilities of pretrained image diffusion models to produce a wide range of visual content without further finetuning. While pretrained image diffusion models cannot directly generate target visual content, most visual content can be converted into a regular image of a specific size through certain mappings. Diffusion synchronization employs such a bridging function between each type of visual content and fixed-sized images. 

In this assignment, we will focus on two applications of diffusion synchronization: arbitrary-sized image and ambiguous image generation generation. We denote the canonical space $\mathcal{Z}$ (e.g., arbitrary-sized images) as the space where a pretrained diffusion model is not provided, and instance spaces as the spaces where pretrained diffusion models are trained (e.g., fixed-size images). Given these two spaces, the projection operation $f$ maps a subset of the canonical space to the instance space, and the unprojection operation $g$ performs the inverse of the projection operation. We will provide the details of the projection and unprojection operations for each application in later sections.

## Setup
Use the same environment as [Assignment 4](https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment4-Distillation).

## Task 0: Introduction 
Recall the reverse step of denoising process presented in [Assignment 1](https://github.com/KAIST-Visual-AI-Group/CS492-Assignment_Diffusion-Models):

$$
\begin{align*}
    q \left( \mathbf{x}^{(t-1)} | \mathbf{x}^{(t)}, \mathbf{x}^{(0)} \right) = \mathcal{N} \left(\psi^{(t)}(\mathbf{x}^{(t)}, \mathbf{x}^{(0)}), \mathbf{I}\right), \quad \text{where} \\
    \psi^{(t)}(\mathbf{x}^{(t)}, \mathbf{x}^{(0)}) = \sqrt{\alpha_{t-1}} \mathbf{x}^{(0)} + \sqrt{\frac{1 - \alpha_{t-1}}{1 - \alpha_t}} \cdot (\mathbf{x}^{(t)} - \sqrt{\alpha_t}\mathbf{x}^{(0)}).
\end{align*}
$$ 

We use fully deterministic sampling for all timestep $t$.

During denoising process, to sample $\mathbf{x}^{(t-1)}$ from its unknown original clean data point $\mathbf{x}^{(0)}$, we estimate $\mathbf{x}^{(0)}$ using Tweedie's formula:

$$
\begin{align*}
    \mathbf{x}^{(0)} \simeq \phi^{(t)}(\mathbf{x}^{(t)}, \epsilon_\theta(\mathbf{x}^{(t)})) 
    &= \frac{\mathbf{x}^{(t)} - \sqrt{1-\alpha_t} \epsilon_\theta(\mathbf{x}^{(t)})}{\sqrt{\alpha_{t}}},
\end{align*}
$$

Arbitrary-sized image generation uses latent-space diffusion model, [Stable Diffusion](https://arxiv.org/abs/2112.10752), and ambiguous image generation uses pixel-space diffusion model, [DeepFloyd IF](https://github.com/deep-floyd/IF). We provide `compute_noise_preds()` method for each diffusion model which corresponds to $\epsilon_\theta(\mathbf{x}^{(t)})$. Your task is to implement basic functionalities of reverse step, `compute_tweedie()` and `compute_prev_state()` in `guidance/base_model.py` which corresponds to $\phi^{(t)}(\cdot, \cdot)$ and $\psi^{(t)}(\cdot, \cdot)$. 

We provide `one_step_process()` in `guidance/base_model.py`, which synchronizes $`\mathbf{x}^{(0)}`$ during the denoising process. However, there is more than one way to synchronize multiple denoising processes which we leave as an optional task. Here, we will focus on SyncTweedie's approach, which aims to synchronize the outputs of Tweedie's formula $`\mathbf{x}^{(0)}`$.  As discussed previously, the synchronization operation requires mapping functions between the instance spaces and the canonical space, which may differ from one application to another. Your task is to implement the mapping functions `inverse_mapping()` and `forward_mapping()` for each application. 

## Task 1: Arbitrary-Sized Image Generation

<p align="center">
<img width="768" alt="image" src="./asset/arbitrary_sized_image_example.png">
</p>

In arbitrary-sized image generation, the canonical space $\mathcal{Z}$ is the space of the arbitrary-sized image with resolution $1 \times C \times H \times W$ ($1 \times 3 \times 512 \times 3072$). The instance spaces $\mathcal{W}_{i=1:N}$ are overlapping patches across the arbitrary-sized image, matching the resolution of the images that the pretrained image diffusion model can generate, $N \times 3 \times 512 \times 512$. We will use latent space diffusion model, Stable Diffusion 2, where latent instance space samples have resolutions of $N \times 4 \times 64 \times 64$, and the canonical space sample has resolution of $1 \times 4 \times 64 \times 384$ (Refer to default configurations in `configs/wide_image_config.py`). The projection operation corresponds to the cropping operation, and the unprojection operation pastes instance space images onto the canonical space sample.

### TODO
To generate arbitrary-sized imges run the following command:

```
python main.py --app wide_image --prompt "{$PROMPT}" --tag wide_image --save_dir_now
```

Once you implement `compute_tweedie()` and `compute_prev_state()`, you can check the outputs of unsynchronized samples by commenting out the following lines in `guidance/base_model.py`.

```
# Synchronization using SyncTweedies
z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization
```

However, generating images without synchronization may result in images that are not view-consistent. To perform diffusion synchronization using SyncTweedies, implement `forward_mapping()` and `inverse_mapping()` in `guidance/wide_image_model.py`. First, `inverse_mapping()` pastes instance space samples onto the canonical space sample using the mapper defined in `init_mapper()` in `guidance/wide_image_model.py` and averages the overlapping regions. On the other hand, `forward_mapping()` crops the canonical space sample to match the resolutions of the instance space samples using the same mapper defined in `init_mapper()`.

## Task 2: Ambiguous Image Generation

<div align="center">
  <img src="./asset/output_ambiguous_deer_truck.gif" alt="First GIF" width="45%">
  <img src="./asset/output_ambiguous_horse_dog.gif" alt="Second GIF" width="45%">
</div>

Ambiguous images are images that exhibit different appearances under certain transformations, such as a 90&deg; rotation or flipping. In this assignment, we will focus on 90&deg; rotation. In ambiguous image generation task, both canonical and instance spaces take the form of images with the same dimensionality. Here, we consider one image as the canonical space, and instance space samples are obtained by applying transformations (forward projection) to the canonical space sample. 

### TODO
To generate ambiguous images run the following command:

```
python main.py --app ambiguous_image --prompts '{$CANONICAL_PROMPT}' '{$INSTANCE_PROMPT}' --tag ambiguous_image --save_dir_now
```

For the 90&deg; rotation transformation, implement `view()` and `inverse_view()` in `utils/views/view_rotate.py`. First, `view()` should return the input image rotated 90&deg; clockwise, and `inverse_view()` should return the image rotated 90&deg; counterclockwise.

## (Optional) Task 3: Exploring Other Diffusion Synchronization Cases

As discussed previously, there is more than one way to synchronize multiple denoising processes. For example, we can synchronize the predicted noises $\epsilon(\mathbf{x}^{(t)})$ or the noisy latents $\mathbf{x}^{(t)}$. In this task, we will generate ambiguous images with inner circle rotation using other diffusion synchronization cases. Unlike the 90&deg; rotation, the inner circle rotation affects only the circular region inside the image while the border remains stationary. 

### TODO
To generate ambiguous images using inner circle rotation, run the following command:

```
python main.py --app ambiguous_image --prompts '{$CANONICAL_PROMPT}' '{$INSTANCE_PROMPT}' --views_names identity inner_rotate --tag ambiguous_image_inner_rotate --save_dir_now
```

First, modify `one_step_process()` in `guidance/base_model.py` to synchronize $\epsilon(\mathbf{x}^{(t)})$ and $\mathbf{x}^{(t)}$, respectively. Then, implement `view_inner_rotate()` in `utils/views/view_inner_rotate.py`, which rotates the inner circular parts of the image. Use 90&deg; for inner circle rotation and apply the same prompts as in [Task 2](#task-2-ambiguous-image-generation) to compare the results of the three diffusion synchronization cases:  $\epsilon(\mathbf{x}^{(t)})$, $\mathbf{x}^{(0)}$ and $\mathbf{x}^{(t)}$.

## What to Submit

For evaluation, we will measure the CLIP score of the generated images. CLIP (Contrastive Language-Image Pre-training) is a model that embeds images and texts into a shared embedding space. The CLIP Score measures the similarity between an image and a text description, with higher scores indicating a closer match. 

### Arbitrary-Sized Image Generation - Evaluation

In arbitrary-sized image evaluation, we randomly crop the arbitrary-sized images and use the cropped instance space samples to measure the CLIP score. When cropping the arbitrary-sized images, use the `eval_pos` field provided in `data/wide_image_prompts.json`. Then, place the cropped images in a single directory and ensure that the generated images are named using their prompts with spaces replaced by underscores, followed by the cropped position (e.g., A_bustling_city_skyline_at_night_with_skyscrapers_243.png).

Then run the following command to measure the CLIP score which will create `eval.json` file:

```
python eval.py --fdir1 {$FDIR} --app ambiguous_images
```

### Ambiguous Image Generation - Evaluation

For evaluation, place the generated images in a single directory and make sure the generated images use their prompts with spaces replaced by underscores (e.g., 'A_cat_and_a_dog.png'). 

Then run the following command to measure the CLIP score which will create `eval.json` file:

```
python eval.py --fdir1 {$FDIR} --app ambiguous_images
```

<details>
<summary><b>Submission Item List</b></summary>
</br>

- [ ] Code
- [ ] PDF file
      
**Task 1**
- [ ] CLIP score evaluation `eval.json` (output of `eval.py`)
- [ ] Output results of generated images using the provided prompts

**Task 2**
- [ ] CLIP score evaluation `eval.json` (output of `eval.py`)
- [ ] Output results of generated images using the provided prompts

**(Optional) Task 3**
- [ ] CLIP score evaluation `eval.json` of each diffusion synchronization case (output of `eval.py`)
- [ ] Output results of generated images using the provided prompts of each diffusion synchronization case

</details>

Submit a zip file named `{NAME}_{STUDENT_ID}.zip` containing the implemented codes and generated images. 
Organize the generated images as below and submit the zip file on GradeScope.
```
.
├── ambiguous_images
│   ├── a_lithograph_of_a_table.png                            <---- Canonical space sample #1
│   ├── a_lithograph_of_a_waterfall.png                        <---- Instance space sample #1
│   ├── an_oil_painting_of_a_canyon.png
│   ├── an_oil_painting_of_a_horse.png
│   ├── an_oil_painting_of_a_library.png
│   ├── an_oil_painting_of_a_maine_coon.png
...
│   └── eval.json
└── wide_images
    ├── A_beach_scene_at_sunrise_with_golden_sands.png         <---- Arbitrary-sized image
    ├── A_beach_scene_at_sunrise_with_golden_sands_1005.png    <---- Cropped image #1
    ├── A_beach_scene_at_sunrise_with_golden_sands_1199.png    <---- Cropped image #2 
    ├── A_beach_scene_at_sunrise_with_golden_sands_1669.png
    ├── A_beach_scene_at_sunrise_with_golden_sands_2220.png
    ├── A_beach_scene_at_sunrise_with_golden_sands_2292.png
    ├── A_beach_scene_at_sunrise_with_golden_sands_2300.png
    ├── A_beach_scene_at_sunrise_with_golden_sands_243.png
    ├── A_beach_scene_at_sunrise_with_golden_sands_991.png
    ...
    └── eval.json
```

## Grading
You will receive a zero score if:
* you do not submit,
* your code is not executable in the Python environment we provided, or
* you modify anycode outside of the section marked with `TODO` or use different hyperparameters that are supposed to be fixed as given.

**Your score will incur a 10% deduction for each missing item in the submission item list.**

Task 1 and Task 2 are worth 10 points each, while Task 3 (Optional) is worth 5 points.

CLIP Score | Points (Optional Task)
--- | ---
0.28 ⬆️ | 10 (5)
0.26 ⬆️ | 5 (2.5)
0.26 ⬇️ | 0 (0)

**Failing to reproduce the reported CLIP score will result in a score of zero.**

This assignment is heavily based on [Visual Anagrams](https://arxiv.org/abs/2311.17919), [MultiDiffusion](https://arxiv.org/abs/2302.08113), and [SyncTweedies](https://arxiv.org/abs/2403.14370). You may refer to the repository while working on the tasks below. However, it is strictly forbidden to simply copy, reformat, or refactor the necessary codeblocks when making your submission. You must implement the functionalities on your own with clear understanding of how your code works. As noted in the course website, we will detect such cases with a specialized tool and plagiarism in any form will result in a zero score.

#### Plagiarism in any form will also result in a zero score and will be reported to the university.

## Further Readings 
* [Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models](https://arxiv.org/abs/2311.17919)
* [MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://arxiv.org/abs/2302.08113)
* [SyncTweedies: A General Generative Framework Based on Synchronized Diffusions](https://arxiv.org/abs/2403.14370)
* [DiffCollage: Parallel Generation of Large Content with Diffusion Models](https://arxiv.org/abs/2303.17076)
* [Text-Guided Texturing by Synchronized Multi-View Diffusion](https://arxiv.org/abs/2311.12891)
* [SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions](https://arxiv.org/abs/2306.05178)
