# How to use the dataset

## Instrcution

This readme file is for **DIPA2** (Dataset with image privacy annotations).

Our work is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. All DIPA2 dataset images is extracted from DIPA, whose images come from [LVIS dataset](https://www.lvisdataset.org/) (extended from [COCO dataset](https://cocodataset.org/#home)) and  [OpenImages V6 dataset](https://storage.googleapis.com/openimages/web/index.html). We welcome you to use or expand DIPA2.

All the demographic information is anonymized to protect the privacy of our annotators.

## File structure

- DIPA_lvis_map.csv --- the map from 22 categories of privacy-threatening content we identified with original categories in [LVIS dataset](https://www.lvisdataset.org/) 

- DIPA_openimages_map.csv --- the map from 22 categories of privacy-threatening content we identified with original categories in [OpenImages V6 dataset](https://storage.googleapis.com/openimages/web/index.html)

- oidv6-class-descriptions.csv --- provided by [OpenImages v6](https://storage.googleapis.com/openimages/web/download_v6.html). It maps the the code of category to the name of it. 

- img_annotation_map.json --- the map of file name from images to privacy-oriented annotations.

- annotations.csv --- an easy-to-use csv file that store all annotations 

- images --- the folder store all image data in their original file names provided by OpenImages and LVIS.

- original labels --- original labels from OpenImages and LVIS dataset in unified format.

- annotations --- the folder store all privacy-oriented labels and worker information collected from [CrowdWorks](https://crowdworks.jp/) and [Prolific](https://www.prolific.co/), which are two crowdsourcing platforms.

- ML baseline --- the folder store the baseline machine learning model in our paper. 

- analyzer.py --- the analysis code we used in the paper. You may run it by `python analyzer`

- logistic regression model.R and annotation_wise_regression_table.csv --- the logistic regression model code and csv file we used in the paper.

  
  
  

## Data structure 

### DIPA_lvis_map.csv

The two-column csv are written in DIPA Category, LVIS Category.

One DIPA category is corresponded with multiple categories in LVIS. 

Each LVIS Category is splited by '|'.

### DIPA_openimages_map.csv

The two-column csv are written in DIPA Category, OpenImages Category.

One DIPA category is corresponded with multiple categories in OpenImages. 

Each OpenImages Category is splited by '|'. 

Please note that the OpenImages categories are written in code.

### oidv6-class-descriptions.csv 

The two-column csv are written in  Code, Name.

You may correspond the code with its actual name when dealing annotations from OpenImages.

### img_annotation_map.json

Each element in this json file has the following key-value structure.

`{imgId: {"CrowdWorks": [labelId], "Prolific": [labelId]}}`

***imgId***: the ID of the target image. You can find the image in the folder "images" by adding suffix ".jpg".

***labelId***: the file name of privacy-oriented labels corresponded with the image.

If the key is "CrowdWorks", you can find the file in ./annotations/CrowdWorks/labels/

If the key is "Prolific", you can find the file in ./annotations/Prolific/labels/

If "CrowdWorks" or "Prolific" does not appear, it means no annotator from the specific platform successfully annotated it.

### annotations.csv

The CSV file that scores all annotations about privacy-threatening content.

You may read the corresponding image in the folder `images` by searching the variable`imagePath`.

For each privacy metric, it will be present in a 0-1 array. If the value is 1, it means that the corresponding answer at the index is chosen by annotators (please refer to the details below).

For example, a value of `[0,0,1,0,1,0]` in the metric "*information type*" means the annotator selected "*It tells individual preferences/pastimes*" and "*It tells others\' private/confidential information*" as his/her answer.

The `bbox` is a two-dimensional array containing all corresponding bounding boxes in the given image. Each element is a 4-D vector indicating a bounding box in the format of`[x, y, width, height]`.

We present other data the same as their raw data, so they will be very easy to understand.

### The arrays of answers of each metrics

**frequency** (Single choice)

​		    ["Never",

​			 "Less than once a month",

​			 "Once or more per month",

​             "Once or more per week",

​			 "Once or more per day"]

##### informationType

​			[ "It tells personal information",

​            "It tells location of shooting",

​            "It tells individual preferences/pastimes",

​            "It tells social circle",

​            "It tells others\' private/confidential information",

​            "Other things it can tell" ]

**informativeness** (Single choice)

​		[ "extremely disagree",

​            "moderately disagree",

​            "slightly disagree",

​            "neutral",

​            "slightly agree",

​            "moderately agree",

​            "extremely agree" ]

PS: please note the actual mapping of informativeness is from -3 to 3. We provided the value of informativeness from 0 to 6 for easier access to the array.

##### sharing scope (as photo owner)

​			[ "I won't share it",

​            "Close relationship",

​            "Regular relationship",

​            "Acquaintances",

​            "Public",

​            "Broadcast program",

​            "Other recipients" ]

##### sharing scope (by others)

​			[ "I won't allow them to share it",

​            "Close relationship",

​            "Regular relationship",

​            "Acquaintances",

​            "Public",

​            "Broadcast program",

​            "Other recipients" ]



## Raw data

We provide raw data in our data collection process if you want to obtain not privacy-threatening content too. 

### Annotators Information

Worker information is stord in ./annotations/CrowdWorks/workerinfo/ and ./annotations/Prolific/workerinfo/

Here is an example:

`{"age": "48", "gender": "Other", "nationality": "Japan", "workerId": "59422", "bigfives": {"Extraversion": 3, "Agreeableness": 5, "Conscientiousness": 7, "Neuroticism": 6, "Openness to Experience": 9}}`

***age***: age of the annotator.

***gender***: gender of the annotator.

***nationality***: nationality of the annotator

***workerid***: worker ID of the annotator from the platform we recruited him/her.

***bigfives***: Big-Five personality inventory results. 



### Privacy-oriented labels

The file name is structured in the following rules:

`{imgId}_{workerId}_label.json`

***imgId***: the ID of the target image. You can find the image in the folder "images" by adding suffix ".jpg". 

***workerId***: the ID of worker who created these labels. 

The file is a json file has the following structure for example. 

Here is an example of the annotations.

`{"source": "OpenImages", "workerId": "annotator23", "defaultAnnotation": {"Object 0": {"category": "Object 0", "informationType": "", "informationTypeInput": "", "informativeness": 0, "sharingOwner": [], "sharingOwnerInput": "", "sharingOthers": [], "sharingOthersInput": "", "ifNoPrivacy": true}, "identity": {"category": "identity", "informationType": [1, 0, 0, 0, 0, 0], "informationTypeInput": "", "informativeness": "7", "sharingOwner": [0, 1, 0, 0, 0, 0, 0], "sharingOwnerInput": "", "sharingOthers": [0, 1, 0, 0, 0, 0, 0], "sharingOthersInput": "", "ifNoPrivacy": false}}, "manualAnnotation": {}}`

***source***: the original dataset of the image.

***workerId***: the ID of worker who created these labels. 

***defaultAnnotation***: the annotation corresponded with labels provided by its original dataset. 

Each element in this dictionary has a key-value structure. For example:

`"Book": {"category": "Book", "informationType": "3", "informationTypeInput": "", "sharing": "3", "sharingInput": "", "ifNoPrivacy": false, "informativeness": "5"}`

The key is the category name from it original dataset. 

- ***informationType***: the answer of question "Assuming you want to seek the privacy of the photo owner, what kind of information can this content tell (please select all that apply)?", provided in a 6-D array.
- ***informationTypeInput***: if the answer includes "other types" (i.e. the last value is 1), other types provided by the annotator will be written here.
- ***informativeness***: the rating results of question "How much do you agree that this content would describe or suggest the people associated with this photo (e.g., the owner of this photo or the person in the photo) in respect of what you chose in the previous question? Higher scores mean the more informative the content is.", provided in numeral format from 1 to 7. 
- ***sharingOwner***: the answer of question "Please assume it is a photo related to you, and answer the following questions. Who would you like to share this content to (please select all that apply)?", provided in a 7-D array.
- ***sharingOwnerInput***: if the answer includes "other recipients" (i.e. the last value is 1), other recipients provided by the annotator will be written here.
- ***sharingOthers***: the answer of question "Please assume it is a photo related to you, and answer the following questions. Would you allow the group you selected above to repost this content (please select all that apply)?", provided in a 7-D array.
- ***sharingOthersInput***: if the answer includes "other recipients" (i.e. the last value is 1), other recipients provided by the annotator will be written here.
- ***ifNoPrivacy***: If true, it indicate the annotator did not think it is a privacy-threatening content. So, no need to read the above results.

***manualAnnotation***: the annotation provided by annotators manually.  

Each element in this dictionary has a key-value structure. For example:

`{"0": {"category": "frequent flyer id", "bbox": [402, 123, 169, 69], "informationType": [1, 0, 0, 0, 1, 0], "informationTypeInput": "", "informativeness": "6", "sharingOwner": [1, 0, 0, 0, 0, 0, 0], "sharingOwnerInput": "", "sharingOthers": [1, 0, 0, 0, 0, 0, 0], "sharingOthersInput": ""}}`

The key is the stringified number of the manual annotation. 

Except the elements that default annotations have, manual annotation has two extra elements.

***category***: category of the annotated object by annotators' own descriptions. Please note that some descriptions are written in Japanese. You may need to install lanuage supports to decode them. 

***bbox***: the bounding box surrounded the object , in the format of `[x, y, width, height]`.



### Original Labels

The file name is structured in the following rules:

`{imgId}_label`

Here is an example.

`{"annotations": {"0": {"category": "Object 0", "bbox": [53, 146, 41, 58]}}, "width": 640, "height": 427, "source": "LVIS"}`

It present each label on the target image, and at least contains the following key-value element.

***category***: category name from its original dataset. 

***bbox***: the bounding box surrounded the object , in the format of `[x, y, width, height]`.

***width***: width of the target image.

***height***: height of the target image.

***source***: the name of its original dataset.

## Note

We build up DIPA2 with two levels of categories: a specific category name from the images' original dataset (i.e.., OpenImages, LVIS) and one of the 24 categories (23 categories + Others) summarized by us. However, some mapping between specific categories from original datasets and 24 categories are adjustable. For instance, given the label "horse" from the original dataset, we can either map it to "pet" or "others" in the 24 categories depending on different contexts.

We highly recommend researchers who intend to use this dataset scrutinize the original category names of annotated content first and decide whether to use the 24 categories summarized by us or directly use the original category provided by OpenImages or LVIS. 