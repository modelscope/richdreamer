# Richdreamer Prompt Engineering Basics

The Richdreamer model generates 3d models based on user inputs. Users
can revise their prompts to arrive at an output to their satisfaction.

The Richdreamer model performs best on food or animal objects. Try to generate 
common objects with lots of data attached, so the model can generate a more accurate model. I recommend having some proficiency in using 3d modelling software to be able to modify the outputs to match desired quality (I used Blender).

***All prompts were run on a system that supports multi-GPU operation using 2 NVIDIA Quadro RTX 6000 GPUS.***

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

**Prompt:** "cute kawaii teddy bear, 3d asset"  
**Execution Time:** 2 hr 41 min

<img src=".\user content gallery\teddy-bear.gif" alt="3d model of cute kawaii teddy bear" style="zoom:200%;" />

Though the this asset has a smoother, round look that can be considered "cute" or "cartoon," the colors are not soft and instead shiny. It is possible the model is not able to under the adjective "kawaii" and was consfused how to interpret this. 

With teddy bears, the model keeps generating results where there is a face on the front and backside (another version seen in the project page's results gallery). This could also be because the Stable diffusion model tries to generate a front view, backview, and overhead view of the teddy bear. There might be incorrect or not engough image data on the backside of teddy bears. 

Either way, I would caution against using this object prompt unless planning to use a modelling software to fix the model afterwards. Most cases, the outputs need some sort of modification/refinement. 

## Limitations with Multi-view Approach

The model learns normal and depth distributions, which helps create consistent and accurate 3D representations across multiple views. 

**Prompt:** "latte in mug with heart-shaped foam art"    
**Execution Time:** 2 hr 12 min

<img src=".\user content gallery\latte.gif" alt="3d model of strawberry donut" style="zoom:200%;" />

The model first generates image representations of the object from multiple views. This prompt would be translated as

**Side:** "latte in mug with heart-shaped foam art, side view"  
**Front:** "latte in mug with heart-shaped foam art, front view"  
**Back:** "latte in mug with heart-shaped foam art, back view"  
**Overhead:** "latte in mug with heart-shaped foam art, overhead view"  

Since the model requires all views to match the prompt, it incorrectly applied the foam art surface feature to the front and back views, even though those were areas where it shouldn't logically appear.

This same limitation of the model might be why the teddy bear had a face on both the front and back sides. 

When engineering prompts, it is best practice to not include features that are limited to one side view. Avoid prompts such as "... with [  ] on front."  


## Holes

**Prompt:** "a strawberry donut, 3d asset"    
**Execution Time:** 1 hr 56 min

<img src=".\user content gallery\donut.gif" alt="3d model of strawberry donut" style="zoom:200%;" />

Richdreamer does an excellent job generating food objects. However, the default negative prompts the model will avoid in the generated output are

"ugly, bad anatomy, blurry, pixelated obscurem unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

Because of how the model interprets the negative prompts, it will not generate objects with large holes. The middle of this donut does not have a hole inside of it. It is filled in with small holes showing that it tried to create some deformation in the donut shape. 

<img src=".\user content gallery\front_face_donut.png" alt="3d model of strawberry donut" style="zoom:200%;" />

Because of this, I recommend not choosing objects with large holes if you do not also plan to alter the output using modelling software.


## Colors



## Aerial View

# Persons

# Imaginary/Dreamer Objects

Dreamer objects refer to objects that don't necessarily exist in real life, but instead ask the Richdreamer model to imagine what it would look like if it did exist in real life. 

Example prompts would be:
"a tiger wearing sunglasses and a leather jacket"
"Wedding dress made of tentacles"
"a group of dogs playing poker"

**Prompt:** "origami tiger in a jungle"    
**Execution Time:** 2 hr 17 min

<img src=".\user content gallery\origami-tiger.gif" alt="3d model of an origami tiger" style="zoom:200%;" />

This model videos well, but after examining it in Blender, I noticed the object mesh was not smooth and the folds were not crisp. Detail is sacrificed when creating imaginary objects and there is more possibility things could go wrong/look deformed. 

Though all object outputs have meshes with bumpy/uneven surfaces, outputs for "origami" seemed to be more so. 

Another noticable thing is color. If color is not specified for objects that can have many colors, the Richdreamer model gets creative and chooses its own color. For this prompt, the output was gold and silver in color. 

In comparison, here is how the model does on a realistic tiger. 

**Prompt:** "realistic tiger, 3d asset"    
**Execution Time:** 2 hr 12 min

<img src=".\user content gallery\tiger.gif" alt="3d model of a tiger" style="zoom:200%;" />

As mentioned, Richdreamer excells in generating highly-realistic animals. Adding the adjective "realistic" was used as a preventative measure, since I wanted to edit the model as little as possible afterwards. 

# "Highly detailed" and other Modifiers