# Image Segmentation by using Graph-cut

## test_seg_mask.py
It helps to test the initial mask of foreground and background.

## generate_histogram.py
It generates color histograms which represent foreground and background.

## cut&#46;py  
First segmenting the area of leaves, 
and then compute the area of leaves.

## Requirements
You need to install pygco to run graphic cut efficiently:
https://github.com/Borda/pyGCO

## Tutorial

1. Define appropriate initial mask pixel location by using 'test_seg_mask.py'  
ref_fg = [350, 450, 1800, 1910]  
ref_bg = [0, 240, 0, 2500]  
![alt text](demo\initial_mask.png)

2. Generate color histogram by using 'generate_histogram.py'  
Multiple reigon from multiple images could be selected.  
  
Example of 'generate_histogram.bat' follows this structure.  
ex>  
python generate_histogram.py images\image-1550434545.jpg images\image-1550079998.jpg images\image-1550434545.jpg images\image-1550434545.jpg ^  
--init_fg_masks 350 450 1800 1910 ^  
--init_fg_masks 880 940 1820 1920 ^  
--init_fg_masks 1450 1500 1850 1940 ^  
--init_fg_masks 950 1000 1200 1250 ^  
--init_bg_masks 0 240 0 2500 ^  
--init_bg_masks 250 320 0 500 ^  
--init_bg_masks 0 1900 1420 1750 ^  
--init_bg_masks 1000 1500 1000 1100 ^  
--work_dir ground_data  


3. Segment images based on the color histogram, and then optimize the segmentation mask by using 'cut&#46;.py'
  
Example of 'cut&#46;.bat' follows this structure.  
  
  ex>    
  python cut&#46;py images\image-1550434545.jpg --histograms ground_data\histograms.npy

