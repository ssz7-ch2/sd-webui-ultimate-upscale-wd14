# Ultimate SD upscale with wd1.4 tagger

Majority of the code is borrowed from [ultimate sd upscale](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111) and [wd1.4 tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)

Recommend using higher thresholds from 0.5-0.9 and denoise between 0.3-0.5 (can work at higher denoise but depends on image)

Is most effective when used on images with large differences between parts of the image (e.g. wide shots & landscape images)  
Not much better when it comes to close ups, as the tagger isn't good at tagging close up objects
### Comparison
Ultimate SD upscale        |  Ultimate SD upscale + wd1.4 tagger
:-------------------------:|:-------------------------:
![00828-4189781129 0-masterpiece, best quality, 1girl, looking up, medium hair, black hair, winter clothes, boots, orange coat, snow, snow mountain,-modified](https://user-images.githubusercontent.com/76718358/218207732-87afcd92-0226-4eff-a7e3-db8f3cd20547.jpg) | ![00827-4189781129-snow, masterpiece, best quality, no humans, solo-modified](https://user-images.githubusercontent.com/76718358/218207737-3fd8436e-27d9-4137-b0e8-303d55cd9658.jpg)


## License
All the borrowed code falls under their respective licenses.  
The few lines that aren't borrowed are under AGPL-3.0 license.
