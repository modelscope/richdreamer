# RichDreamer Prompt Engineering Basics

The RichDreamer model generates 3d models based on user inputs. Users
can revise their prompts to arrive at an output to their satisfaction.

**The RichDreamer model performs best on food or animal objects.** Try to generate common objects with a lot of data attached, so the model can generate a more accurate output. I recommend having some proficiency in using 3d modelling software to be able to modify the outputs to match desired quality (I used Blender).

***All prompts were run on a system that supports multi-GPU operation using 2 NVIDIA Quadro RTX 6000 GPUS.***

Runtime for the model is ≈2 hrs. As users revise prompts of the same object, runtime can decrease to as little as ≈1 hr 15 min.
Because execution time is long, it is important to utilize each prompt entry. Here are some things to remember when generating prompts:

**High-Level Advice: It is harder to have a specific output in mind and try to get the RichDreamer model to recreate that. It is better to have general idea of the object you want, then roll with the results.**

## "3d asset" as a Modifier

In the project page's gallery results, there are some prompts with the modifier ", 3d asset." Is this modifier really necessary? 

It is not necessary, but it helps the Stable Diffusion model generate images that can more easily be translated into 3d outputs. 

For example, when asking DALL·E 3 to generate an image of "a cute kawaii teddy bear", it produced images like the one below. 

<img src=".\figs\prompt engineering guide\teddy_bear_dalle3.png" alt="DALL·E 3 image of cute kawaii teddy bear" style="width:40%;" />

Alternatively, when asking the prompt "a cute kawaii teddy bear 3d asset," it generates pictures like the one below. 

<img src=".\figs\prompt engineering guide\teddy_bear_3dasset_dalle3.png" alt="DALL·E 3 image of cute kawaii teddy bear, 3d asset" style="width:40%;" />

As seen, the second can more easily be translated into 3d. Assuming the Stable Diffusion model follows similar prompt engineering principles as DALL·E 3 (since both implement text-to-image generative AI), adding the modifier ", 3d asset" can help ensure the output will be of better quality.

## Different Illustration Styles

RichDreamer is best at handling a realistic generation style. It does not perform well on cartoon/cute/kawaii illustration styles. 

**Prompt:** "cute kawaii teddy bear, 3d asset"  
**Execution Time:** 2 hr 41 min

<img src=".\user content gallery\cute_kawaii_teddy_bear,_3d_asset.gif" alt="3d model of cute kawaii teddy bear" style="zoom:200%;" />

Though the this asset has a smoother, round look that can be considered "cute," the colors are stark. It's possible the model was consfused how to interpret the adjective "kawaii." 

With teddy bears, the model keeps generating results where there is a face on the front and backside (another version seen in the project page's results gallery). This could also be because the Stable diffusion model tries to generate a front view, backview, sideview, and overhead view of the teddy bear. There might be incorrect or not engough image data on the backside of teddy bears. 

Either way, I would caution against using this object prompt unless planning to use a modelling software to fix the model afterwards. Most cases, the outputs need some modification/refinement. 

## Limitations with Multi-view Approach

The RichDreamer model learns normal and depth distributions, which helps create consistent and accurate 3d representations across multiple views. 

**Prompt:** "latte in mug with heart-shaped foam art"    
**Execution Time:** 2 hr 12 min

<img src=".\user content gallery\latte_in_mug_with_heart-shaped_foam_art.gif" alt="3d model of latte in mug with foam art" style="zoom:200%;" />

The model first generates image representations of the object from multiple views then creates a 3d output from the images. This prompt would be translated as

**Side:** "latte in mug with heart-shaped foam art, side view"  
**Front:** "latte in mug with heart-shaped foam art, front view"  
**Back:** "latte in mug with heart-shaped foam art, back view"  
**Overhead:** "latte in mug with heart-shaped foam art, overhead view"  

Since the model requires all views to match the prompt, it incorrectly applied the foam art surface feature to the front and back views, even though those weren't areas they should logically appear.

This same limitation of the model might be why the teddy bear had a face on both the front and back sides. 

When engineering prompts, it is best practice to not include features that are limited to one side view. If necessary,  try to be specific where the feature is located. This prompt might have performed better if it was "... on top."  


## Holes

**Prompt:** "a strawberry donut, 3d asset"    
**Execution Time:** 1 hr 56 min

<img src=".\user content gallery\a_strawberry_donut,_3d_asset.gif" alt="3d model of strawberry donut" style="zoom:200%;" />

RichDreamer does an excellent job generating food objects. However, the default negative prompts the model will avoid in the generated output are

"ugly, bad anatomy, blurry, pixelated obscurem unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"

Because of how the model interprets the negative prompts, it will not generate objects with large holes. The middle of this donut does not have a hole inside of it. The middle has some small holes showing the model tried to create some deformation in the donut shape. 

<img src=".\user content gallery\front_face_donut.png" alt="front view of strawberry donut 3d model with no hole in middle" style="zoom:200%;" />

I recommend not choosing objects with large holes unless planning to alter the output using modelling software.


## Colors

Specifying colors of objects can be a good way to guide the model to create the output in mind. 

**Prompt:** "light blue polaroid camera, 3d asset"    
**Execution Time:** 2 hr 39 min

<img src=".\user content gallery\light_blue_polaroid_camera,_3d_asset.gif" alt="3d model of light blue polaroid camera" style="zoom:200%;" />

When guiding the model to create outputs with certain colors, I found the color clarity is not as great as when the model chooses the colors itself. This is most likely because of specific colors used in the dataset images.  For example, the polaroid camera's colors lacked sharpness and the boundaries were not well defined when examining the model in Blender. 

**Prompt:** "light blue flower vase with red orange tulips"    
**Execution Time:** 2 hr 17 min

<img src=".\user content gallery\light_blue_flower_vase_with_red_orange_tulips.gif" alt="3d model of blue flower vase with red orange tulips" style="zoom:200%;" />

The flower vase had vibrant colors, but the blue from the flower vase spilled into the green stems, which wasn't ideal. Also, the mesh was not detailed, which I found was a problem with flower/tree objects for the RichDreamer model. 

**Prompt:** "vintage white convertible car, 3d asset"    
**Execution Time:** 3 hr 15 min

<img src=".\user content gallery\vintage_white_convertible_car,_3d_asset.gif" alt="3d model of white vintage convertible car" style="zoom:200%;" />

The car's mesh was also not detailed. It was bumpy and deformed when inspecting in Blender, and the model added gold detailing to match the "vintage" prompt description. 

The model might add more colors to match other parts of the prompt.

RichDreamer performs better when choosing its own colors, where the object mesh will be more detailed and colors will have more sharpness. However, because it can be hard to change the texture colors by hand, users might sacrifice this quality when generating outputs.  

## Aerial View

The RichDreamer model works well on aerial view landscapes, which can be **beneficial for CGI and drone footage applications.**

In the project page's gallery results, users can see the model creates exceptional models of Mont Saint-Michel, France in aerial view and Neuschwanstein Castle in aerial view. The model has knowledge of global landmarks, letting users create outputs like these. 

Because the view is aerial, the need for precise detail is not necessary, so the model performs well. These outputs video well, making AI-generated 3d great for CGI. 

**Prompt:** "island, highly detailed, aerial view"    
**Execution Time:** 2 hr 15 min

<img src=".\user content gallery\island,_highly_detailed,_aerial_view.gif" alt="3d model of a tropical island" style="zoom:200%;" />

When it comes to aerial objects, however, it is unpredicable the amount of background/setting the model will output. For example, the model outputted no ocean or water when that would have been more ideal. To be safe, I should have included in the prompt "island in middle of ocean..." 

**Prompt:** "an erupting volcano, aerial view"    
**Execution Time:** 2 hr 21 min

<img src=".\user content gallery\an_erupting_volcano,_aerial_view.gif" alt="3d model of an erupting volcano" style="zoom:200%;" />

The boundaries of aerial objects are usually jagged. The model can output a lot of land area or just include the main object. For this volcano, there is some volcanic rock land area.

**Prompt:** "an erupting volcano with lava and smoke coming out, aerial view"    
**Execution Time:** 3 hr 15 min

<img src=".\user content gallery\an_erupting_volcano_with_lava_and_smoke_coming_out,_aerial_view.gif" alt="3d model of an erupting volcano with more smoke and lava" style="zoom:200%;" />

In this volcano output, the boundary is much more uneven.

In my mind, I wanted big clouds of smoke billowing out of the volcano (like in the project page gallery). However, even after modifying the prompt, this did not happen.

As the model trains on revised prompts, the output becomes more detailed, but most time the shape or object does not drastically change. 

**The model adapts its interpretation of the object based on new prompts and does not start training over again. Weights are adjusted on prompt retries, so to create the same object with a drastically different look, model weights would need to be reset.**


## Persons

**Prompt:** "cartoon pink fairy princess, 3d character"    
**Execution Time:** 2 hr 18 min

<img src=".\user content gallery\cartoon_pink_fairy_princess,_3d_character.gif" alt="3d model of a pink fairy princess" style="zoom:200%;" />

It seems that the model interpretted "cartoon" as including  solid-color features for the fairy. "Princess" is not shown in any way, so adding too many references in a prompt can lead to one being missed. 

The character has pink hair, dress, wings, and shoes. The model can place specified colors anywhere, even in unexpected places. I recommend specifying where the color is placed, like if I modified the prompt as "cartoon fairy princess in a pink dress..."

**Prompt:** "a ballerina figureine with a purple tutu, 3d character"    
**Execution Time:** 2 hr 41 min

<img src=".\user content gallery\a_ballerina_figurine_with_a_purple_tutu,_3d_character.gif" alt="3d model of a ballerina" style="zoom:200%;" />

This ballerina has a clear deformation in the face. Also, it is missing a leg, which might be because of certain dataset images where the leg is hidden in the ballet pose. 

It seems the model interprets "3d character" to have a less reailistic, more doll-like apprearance.

**Prompt:** "a ballerina figurine with a purple tutu"    
**Execution Time:** 2 hr 21 min

<img src=".\user content gallery\a_ballerina_figurine_with_a_purple_tutu.gif" alt="more detailed 3d model of a ballerina" style="zoom:200%;" />

Upon revising the prompt, we get a more detailed, realistic output. The leg is now added after the model had time to reevaluate its results. The tutu and he skirt are also more attached. However, the output contains the same deformation in the face as the original, and it was too servere to fix up on Blender.

I recommend not creating people, because it can be too risky with deformations. Users might need to run the model multiple times and/or modify the output with a modelling software. Keeping the prompt vague can help RichDreamer apply what it has the most data on.  

## Imaginary/Dreamer Objects

Dreamer objects refer to objects that don't necessarily exist in real life, but instead ask the RichDreamer model to imagine what they would look like if they did exist in real life. 

Example prompts would be:  
"a tiger wearing sunglasses and a leather jacket"  
"Wedding dress made of tentacles"  
"a group of dogs playing poker"

**Prompt:** "origami tiger in a jungle"    
**Execution Time:** 2 hr 17 min

<img src=".\user content gallery\origami_tiger_in_jungle.gif" alt="3d model of an origami tiger" style="zoom:200%;" />

This model videos well, but after examining it in Blender, I noticed the object mesh was not smooth and the folds were not crisp. Detail is sacrificed when creating imaginary objects and there is more possibility things could go wrong/look deformed. 

Though all object outputs have meshes with bumpy/uneven surfaces, outputs for "origami" seemed to be more so. 

Another noticable thing is color. If color is not specified for objects, the RichDreamer model gets creative and chooses its own color. For this prompt, the tiger output was gold and silver. 

In comparison, here is how the RichDreamer model does on a realistic tiger. 

**Prompt:** "realistic tiger, 3d asset"    
**Execution Time:** 2 hr 12 min

<img src=".\user content gallery\realistic_tiger,_3d_asset.gif" alt="3d model of a realistic tiger" style="zoom:200%;" />

As mentioned, RichDreamer excels in generating highly-realistic animals. Adding the adjective "realistic" was used as a preventative measure, since I wanted to edit the model as little as possible afterwards. 

# "Highly detailed" and other Modifiers

**Prompt:** "a china teapot, 3d asset"    
**Execution Time:** 1 hr 37 min

<img src=".\user content gallery\a_china_teapot,_3d_asset.gif" alt="3d model of a china teapot" style="zoom:200%;" />

**Prompt:** "a white and blue china teapot, highly detailed, 3d asset"    
**Execution Time:** 1 hr 40 min

<img src=".\user content gallery\a_white_and_blue_china_teapot,_highly_detailed,_3d_asset.gif" alt="3d model of a blue and white china teapot" style="zoom:200%;" />

Upon the prompt retry with "highly detailed" and adding "white and blue" colors, the texture had more intricate patterns. The increase in detail could also be because it was the second prompt iteration.

The modifier "highly detailed" can be used as precautionary measure so the model tries harder to create an output of high quality. 

The color was silver instead of white. This is because the model adjusted weights from the original teapot, which was gold (and silver is closer to white than gold). As mentioned before, prompt retries cannot drastically change the model ouput; it can only refine it. 

I recommend being detailed on the intial prompt entry to include colors and modifiers like"highly detailed" then continue training the model to increase quality. 

## Text-to-3d Applications

As text-to-3d generative AI models improve in capability, they can be used for customization in CGI and game development. 

**Film and Animation:** Text-to-3d models can be used to quickly generate high-quality 3d assets for movies, TV shows, and animated films. Instead of manually modeling every object, artists can describe what they need, and AI generates detailed outputs that can be further refined or directly used in scenes. This can significantly speed up the production process and reduce costs.

**Game Asset Creation:** Game developers can use text-to-3d models to generate assets such as characters, environments, and props based on simple text prompts. This allows for quicker iterations and creativity in game design. Developers can experiment with different ideas without needing extensive manual modeling.

**Advertising and Marketing:** In advertising, these models can create 3d visuals for products and environments based on textual descriptions, allowing for rapid prototyping and customization. This is particularly useful for creating photorealistic images and videos that can adapt to different campaigns and client needs.

## Gallery

Below are animations of some of the RichDreamer 3d outputs! I edited the outputs and produced the animations in Blender. 


**Prompt:** "a strawberry donut, 3d asset"    
**Execution Time:** 1 hr 56 min  
<img src=".\user content gallery\strawberry_donut_animation.gif" alt="camera spin animation of strawberry donut 3d model" style="zoom:200%;" />

**Prompt:** "fresh fig cut in half, 3d asset"    
**Execution Time:** 3 hr 49 min  
<img src=".\user content gallery\fig_animation.gif" alt="camera spin animation of fig 3d model" style="zoom:200%;" />

**Prompt:** "realistic tiger, 3d asset"    
**Execution Time:** 2 hr 12 min  
<img src=".\user content gallery\realistic_tiger_animation.gif" alt="camera spin animation of realistic tiger 3d model" style="zoom:200%;" />

**Prompt:** "origami tiger in jungle"    
**Execution Time:** 2 hr 17 min  
<img src=".\user content gallery\origami_tiger_animation.gif" alt="camera spin animation of origami tiger 3d model" style="zoom:200%;" />

**Prompt:** "stuffed animal dog with bow around neck"    
**Execution Time:** 2 hr 20 min  
<img src=".\user content gallery\stuffed_dog_animation.gif" alt="camera spin animation of stuffed dog 3d model" style="zoom:200%;" />

**Prompt:** "cherry blossom tree"    
**Execution Time:** 1 hr 28 min  
<img src=".\user content gallery\cherry_blossom_animation.gif" alt="camera spin animation of stuffed dog 3d model" style="zoom:200%;" />
