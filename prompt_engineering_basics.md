# Richdreamer Prompt Engineering Basics

The Richdreamer model generates 3d models based on user inputs. Users
can revise their prompts to arrive at an output to their satisfaction.

The Richdreamer model performs best on food or animal objects. Try to generate 
common objects with lots of data attached, so the model can generate a more accurate model. I recommend having some proficiency in using 3d modelling software to be able to modify the outputs to match desired quality (I used Blender).

Runtime for the model is usually ≈2 hrs. As you revise prompts of the same object, runtime can decrease to as little as ≈1 hr 15 min.
Because execution time is long, it is important to utilize each prompt entry. Here are some things to remember when generating prompts:

## "3d asset" as a Modifier

## Holes

**Prompt:** "a strawberry donut, 3d asset" \
**Execution Time:** 1 hr 56 min

<img src=".\user content gallery\donut.gif" alt="3d model of strawberry donut" style="zoom:200%;" />

Richdreamer does an excellent job generating food objects. However, the default negative prompts the model will avoid in the generated output are

"ugly, bad anatomy, blurry, pixelated obscurem unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

Because of how the model interprets the negative prompts, it will not generate objects with large holes. The middle of this donut does not have a hole inside of it. It is filled in with small holes showing that it tried to create some deformation in the donut shape. 

<img src=".\user content gallery\front_face_donut.png" alt="3d model of strawberry donut" style="zoom:200%;" />

Because of this, I reccommend not choosing objects with large holes if you require the 


## Colors

## Aerial View

# Persons

# "Highly detailed" and other Modifiers