# Richdreamer Prompt Engineering Basics

The Richdreamer model generates 3d models based on user inputs. Users
can revise their prompts to arrive at an output to their satisfaction.

**The Richdreamer model performs best on food or animal objects.** Try to generate common objects with lots of data attached, so the model can generate a more accurate model. I recommend having some proficiency in using 3d modelling software to be able to modify the outputs to match desired quality (I used Blender).

***All prompts were run on a system that supports multi-GPU operation using 2 NVIDIA Quadro RTX 6000 GPUS.***

Runtime for the model is usually ≈2 hrs. As you revise prompts of the same object, runtime can decrease to as little as ≈1 hr 15 min.
Because execution time is long, it is important to utilize each prompt entry. Here are some things to remember when generating prompts:

**High-Level Advice: It is harder to have a specific output in mind and try to get the Richdreamer model to recreate that. It is better to have general idea of the object you want, then roll with the results.**

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

Specifying colors of objects can be a good way to guide the model to create the output you have in mind. 

**Prompt:** "light blue polaroid camera, 3d asset"    
**Execution Time:** 2 hr 39 min

<img src=".\user content gallery\camera.gif" alt="3d model of light blue polaroid camera" style="zoom:200%;" />

When guiding the model to create outputs with certain colors, I found the color clarity is not as great as when the model chooses the colors itself. This is most likely because of the specific colors from the dataset images.  For example, the polaroid camera's colors lacked sharpness, as in the boundaries were not well defined when examining the model in Blender. 

**Prompt:** "light blue flower vase with red orange tulips"    
**Execution Time:** 2 hr 17 min

<img src=".\user content gallery\flower_vase.gif" alt="3d model of blue flower vase with red orange tulips" style="zoom:200%;" />

The flower vase had vibrant colors, but the blue from the flower vase spilled into the green stems, which wasn't ideal. Also, the mesh was not detailed, which I found was a problem with flower/tree objects for the Richdreamer model. 

**Prompt:** "vintage white convertible car, 3d asset"    
**Execution Time:** 3 hr 15 min

<img src=".\user content gallery\car.gif" alt="3d model of white vintage convertible car" style="zoom:200%;" />

The car's mesh was also not detailed. It was bumpy and deformed when seeing in Blender, and the model added gold detailing to match the "vintage" prompt description. 

Richdreamer performs better when choosing its own colors, where the object mesh will be more detailed and colors will have more sharpness. However, because it can be hard to change the texture colors by hand, you might need to sacrifice this quality when generating outputs.  

## Aerial View

The Richdreamer model works well on aerial view landscapes, which can be **beneficial for CGI and drone footage applications.**

In the project page's gallery results, you can see the model creates exceptional models of Mont Saint-Michel, France in aerial view and Neuschwanstein Castle in aerial view. The model has knowledge of global landmarks, letting you create outputs like these. 

Because the view is aerial, the need for precise detail is not necessary, so the model performs well. These outputs video well, making AI-generated 3d great for CGI. 

**Prompt:** "island, highly detailed, aerial view"    
**Execution Time:** 2 hr 15 min

<img src=".\user content gallery\island.gif" alt="3d model of a tropical island" style="zoom:200%;" />

When it comes to aerial objects, however, it is unpredicable the amount of background setting the model will output. For example, the model outputted no ocean or water when that would have been more logical. 

**Prompt:** "an erupting volcano, aerial view"    
**Execution Time:** 2 hr 21 min

<img src=".\user content gallery\volcano.gif" alt="3d model of an erupting volcano" style="zoom:200%;" />

The boundaries of aerial objects are also usually jagged. The model can out lots of land area or just include the main object. For this volcano, there is some volcanic rock on land.

**Prompt:** "an erupting volcano with lava and smoke coming out, aerial view"    
**Execution Time:** 3 hr 15 min

<img src=".\user content gallery\volcano2.gif" alt="3d model of an erupting volcano with smoke and lava" style="zoom:200%;" />

In this volcano output, the boundary is much more uneven.

In my mind, I wanted big clouds of smoke billowing out of the volcano, like in the project page gallery. However, even after modifying the prompt, this did not happen.

As the model trains on revised prompts, the output becomes more detailed, but most time the shape or object does not drastically change. 

The model adapts its interpretation of the object based on new prompt and does not start training over again. Weights are adjusted on prompt retries, so to create the same object with a drastically different look, model weights might need to be reset.


## Persons

## Imaginary/Dreamer Objects

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