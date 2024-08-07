# Richdreamer Prompt Engineering Basics

The Richdreamer model generates 3d models based on user inputs. Users
can revise their prompts to arrive at an output to their satisfaction.

The Richdreamer model performs best on food or animal objects. Try to generate 
common objects with lots of data attached, so the model can generate a more accurate model. I recommend having some proficiency in using 3d modelling software to be able to modify the outputs to match desired quality (I used Blender).

Runtime for the model is usually ≈2 hrs. As you revise prompts of the same object, runtime can decrease to as little as ≈1 hr 15 min.
Because execution time is long, it is important to utilize each prompt entry. Here are some things to remember when generating prompts:

## "3d asset" as a Modifier

In the project page's gallery results, there are some prompts with the modifier ", 3d asset." Is this modifier really necessary? 

It is not necessary, but it helps the Stable Diffusion model generate images that can more easily be translated into 3d outputs. 

For example, when asking DALL·E 3 to generate an image of "a cute kawaii teddy bear", it produced images like the one below. 

<img src=".\figs\prompt engineering guide\teddy_bear_dalle3.png" alt="DALL·E 3 image of cute kawaii teddy bear" style="zoom:40%;" />

Alternatively, when asking the prompt "a cute kawaii teddy bear 3d asset," it generates pictures like the one below. 

<img src=".\figs\prompt engineering guide\teddy_bear_3dasset_dalle3.png" alt="DALL·E 3 image of cute kawaii teddy bear, 3d asset" style="zoom:40%;" />

As seen, the second can more easily be translated into 3d. Assuming the Stable Diffusion model follows similar prompt engineering principles (since both implement text-to-image generative AI), adding the modifier ", 3D asset" can help ensure the output will be of better quality.

## Different Illustration Styles

Richdreamer is best at handling a realistic generation style. It is not perform well on cartoon/cute/kawaii illustration styles. 

**Prompt:** "cute kawaii teddy bear, 3d asset" \
**Execution Time:** 2 hr 41 min

<img src=".\user content gallery\teddy-bear.gif" alt="3d model of cute kawaii teddy bear" style="zoom:200%;" />

Though the this asset has a smoother, round look that can be considered "cute" or "cartoon," the colors are not soft and instead shiny. It is possible the model is not able to under the adjective "kawaii" and was consfused how to interpret this. 

With teddy bears, the model keeps generating results where there is a face on the front and backside (another version seen in the project page's results gallery).

## Holes

**Prompt:** "a strawberry donut, 3d asset" \
**Execution Time:** 1 hr 56 min

<img src=".\user content gallery\donut.gif" alt="3d model of strawberry donut" style="zoom:200%;" />

Richdreamer does an excellent job generating food objects. However, the default negative prompts the model will avoid in the generated output are

"ugly, bad anatomy, blurry, pixelated obscurem unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

Because of how the model interprets the negative prompts, it will not generate objects with large holes. The middle of this donut does not have a hole inside of it. It is filled in with small holes showing that it tried to create some deformation in the donut shape. 

<img src=".\user content gallery\front_face_donut.png" alt="3d model of strawberry donut" style="zoom:200%;" />

Because of this, I recommend not choosing objects with large holes if you do not plan to alter the output using modelling software.


## Colors

## Aerial View

# Persons

# "Highly detailed" and other Modifiers