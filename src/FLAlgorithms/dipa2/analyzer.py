import os
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class analyzer:
    def __init__(self) -> None:
        self.annotation_path = './annotations/'
        self.platforms = ['CrowdWorks', 'Prolific']
        self.img_annotation_map_path = './img_annotation_map.json'
        self.img_annotation_map = {}
        self.code_openimage_map = {}
        self.openimages_mycat_map = {}
        self.lvis_mycat_map = {}
        self.test_size = 0.2
        self.custom_informationType = []
        self.custom_recipient_owner = []
        self.custom_recipient_others = []
        self.description = {'informationType': ['personal information', 'location of shooting',
        'individual preferences/pastimes', 'social circle', 'others\' private/confidential information', 'Other things'],
        'informativeness':['Strongly disagree','Disagree','Slightly disagree','Neither',
        'Slightly agree','Agree','Strongly agree'],
        'sharingOwner': ['I won\'t share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'], 
        'sharingOthers':['I won\'t allow others to share it', 'Close relationship',
        'Regular relationship', 'Acquaintances', 'Public', 'Broadcast program', 'Other recipients'],
        'frequency': ['Never', 'Less than once a month', 'Once or more per month', 
        'Once or more per week', 'Once or more per day']}
        self.mega_table_path = './annotations.csv'
        if not os.path.exists(self.img_annotation_map_path):
            self.generate_img_annotation_map()
        with open(self.img_annotation_map_path) as f:
            self.img_annotation_map = json.load(f)
        with open('./DIPA_lvis_map.csv') as f:
                res = csv.reader(f)
                flag = 0
                for row in res:
                    if not flag:
                        flag = 1
                        continue
                    lvis_cats = row[1].split('|')
                    if 'None' in row[1]:
                        continue
                    for cat in lvis_cats:
                        self.lvis_mycat_map[cat] = row[0]
        with open('./oidv6-class-descriptions.csv') as f:
            res = csv.reader(f)
            for row in res:
                self.code_openimage_map[row[0]] = row[1]
        with open('./DIPA_openimages_map.csv') as f:
            res = csv.reader(f)
            flag = 0
            for row in res:
                if not flag:
                    flag = 1
                    continue
                openimages_cats = row[1].split('|')
                if 'None' in row[1]:
                    continue
                for cat in openimages_cats:
                    category_name = self.code_openimage_map[cat]
                    self.openimages_mycat_map[category_name] = row[0]

    def basic_info(self, platform='CrowdWorks')->None:
        age = {'18-24': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '25-34': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '35-44': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '45-54': {'Male': 0, 'Female': 0, 'Other': 0}, 
        '55': {'Male': 0, 'Female': 0, 'Other': 0}}
        info_paths = os.listdir(os.path.join(self.annotation_path, platform, 'workerinfo'))
        for info_path in info_paths:
            # check if nan, nan!=nan
            with open(os.path.join(self.annotation_path, platform, 'workerinfo', info_path)) as f:
                text = f.read()
                info = json.loads(text)
                if int(info['age']) != int(info['age']):
                    print('wrong age found', info)
                    continue
                year = int(info['age'])
                if year < 18:
                    print('wrong age found', info)
                if 18 <= year <= 24:
                    age['18-24'][info['gender']] += 1
                elif 25 <= year <= 34:
                    age['25-34'][info['gender']] += 1
                elif 35 <= year <= 44:
                    age['35-44'][info['gender']] += 1
                elif 45 <= year <= 54:
                    age['45-54'][info['gender']] += 1
                elif 55 <= year:
                    age['55'][info['gender']] += 1
        print(age)

            
    def distribution(self, strict_mode = False, strict_num = 2)->None:
        #distribution of privacy and not privacy annotations in each category
        #if it is privacy, then calculate how many times for each object is annotated as privacy

        ## calculate amount of all category 
        category_and_id = {}
        category_number = {}
        for image_name in self.img_annotation_map.keys():
            for platform, annotations in self.img_annotation_map[image_name].items():
                for i, annotation in enumerate(annotations):
                    if strict_mode and i >= strict_num:
                        break
                    with open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f:
                        label = json.load(f)
                        dataset_name = label['source']
                        for key, value in label['defaultAnnotation'].items():
                            if dataset_name == 'OpenImages':
                                if key in self.openimages_mycat_map.keys():
                                    category = self.openimages_mycat_map[key]
                                else:
                                    category = 'others'
                            elif dataset_name == 'LVIS':
                                if key in self.lvis_mycat_map.keys():
                                    category = self.lvis_mycat_map[key]
                                else:
                                    category = 'others'
                            object_id = image_name + '_' + key + '_' + category
                            if category not in category_and_id.keys():
                                category_and_id[category] = {}
                            category_and_id[category][object_id] = 0

        for key, value in category_and_id.items():
            category_number[key] = len(value)
        
        # add up category number
        tot_num = 0
        for key, value in category_number.items():
            tot_num += value

        print(category_number)
        print('total:', tot_num)

        # calculate amount of privacy category
        privacy_category = {'CrowdWorks':{}, 'Prolific':{}, 'All': {}}
        privacy_num = {'CrowdWorks':{}, 'Prolific':{}, 'All': {}}

        self.mega_table = pd.read_csv(self.mega_table_path)


        exclude_manual = self.mega_table[self.mega_table.category != 'Manual Label']
        # access each row of exclude_manual
        for index, row in exclude_manual.iterrows():
            dataset_name = row['originalDataset']
            platform = row['platform']
            key = row['category']
            image_name = row['imagePath'][:-4]
            if dataset_name == 'OpenImages':
                if key in self.openimages_mycat_map.keys():
                    category = self.openimages_mycat_map[key]
                else:
                    category = 'others'
            elif dataset_name == 'LVIS':
                if key in self.lvis_mycat_map.keys():
                    category = self.lvis_mycat_map[key]
                else:
                    category = 'others'
            object_id = image_name + '_' + key + '_' + category
            if category not in privacy_category[platform].keys():
                privacy_category[platform][category] = {}
            if category not in privacy_category['All'].keys():
                privacy_category['All'][category] = {}

            if object_id not in privacy_category[platform][category].keys():
                privacy_category[platform][category][object_id] = 1
            else:
                privacy_category[platform][category][object_id] += 1

            if object_id not in privacy_category['All'][category].keys():
                privacy_category['All'][category][object_id] = 1
            else:
                privacy_category['All'][category][object_id] += 1
        
        for platform, category in privacy_category.items():
            for key, value in category.items():
                privacy_num[platform][key] = {1: 0, 2: 0, 3: 0, 4: 0}
                for object_id, num in value.items():
                    privacy_num[platform][key][num] += 1
        
        #Add up privacy number of All
        tot = {'CrowdWorks': {1: 0, 2: 0, 3: 0, 4: 0},
                'Prolific': {1: 0, 2: 0, 3: 0, 4: 0},
                'All': {1: 0, 2: 0, 3: 0, 4: 0}}
        for platform in privacy_num.keys():
            for key, value in privacy_num[platform].items():
                if key == 'others':
                    continue
                tot[platform][1] += value[1]
                tot[platform][2] += value[2]
                tot[platform][3] += value[3]
                tot[platform][4] += value[4]

        print('CrowdWorks:')
        print(privacy_num['CrowdWorks'])
        print('Prolific:')
        print(privacy_num['Prolific'])
        print('All:')
        print(privacy_num['All'])

        print('total:', tot)

    def count_frequency(self)->None:
        ## count frenquency of sharing for each annotator
        frequency = {'CrowdWorks': {0: 0, 1:0, 2:0, 3:0, 4:0}, 'Prolific': {0: 0, 1:0, 2:0, 3:0, 4:0}, 'All': {0: 0, 1:0, 2:0, 3:0, 4:0}}
        # CrowdWorks
        worker_file = os.listdir('./annotations/CrowdWorks/workerinfo')
        for file in worker_file:
            with open('./annotations/CrowdWorks/workerinfo/' + file) as f:
                worker = json.load(f)
                frequency['CrowdWorks'][int(worker['frequency'])] += 1
                frequency['All'][int(worker['frequency'])] += 1
        # Prolific
        worker_file = os.listdir('./annotations/Prolific/workerinfo')
        for file in worker_file:
            with open('./annotations/Prolific/workerinfo/' + file) as f:
                worker = json.load(f)
                frequency['Prolific'][int(worker['frequency'])] += 1
                frequency['All'][int(worker['frequency'])] += 1
        
        print(frequency)

    def non_max_suppression(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding boxes by their area
        area = boxes[:, 2] * boxes[:, 3]
        idxs = np.argsort(area)[::-1]

        # loop over the indexes of the bounding boxes
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the bounding box and the rest of the bounding boxes
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have an overlap greater than the overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        return boxes[pick]

    def basic_count(self, split_count = False, count_scale = 'CrowdWorks',
                    strict_mode = True, ignore_prev_manual_anns=False, strict_num = 2) -> None:

        def calculate_array(input_array, option_num):
            res = np.zeros(option_num, dtype='int')
            for i in range(input_array.shape[0]):
                res += np.array(json.loads(input_array[i]))
            return res

        self.mega_table = pd.read_csv(self.mega_table_path)

        if split_count:
            print(self.mega_table)
            self.mega_table = self.mega_table[self.mega_table['platform'] == count_scale]
            print(self.mega_table)
        frequency = self.mega_table['frequency'].value_counts()
        frequency = frequency.sort_index().values
        frequency = pd.DataFrame([frequency], columns=self.description['frequency'])
        informationType = calculate_array(self.mega_table['informationType'].values, 6)
        informationType = pd.DataFrame([informationType], columns=self.description['informationType'])
        informativeness = self.mega_table['informativeness'].value_counts()
        print(informativeness)
        informativeness = informativeness.sort_index().values
        informativeness = pd.DataFrame([informativeness], columns=self.description['informativeness'])

        #informativeness = pd.DataFrame([informativeness], columns=self.description['informativeness'])
        sharingOwner = calculate_array(self.mega_table['sharingOwner'].values, 7)
        sharingOwner = pd.DataFrame([sharingOwner], columns=self.description['sharingOwner'])
        sharingOthers = calculate_array(self.mega_table['sharingOthers'].values, 7)
        sharingOthers = pd.DataFrame([sharingOthers], columns=self.description['sharingOthers'])

        print('----------{}----------'.format('frequency'))
        print(frequency)
        print('----------{}----------'.format('informationType'))
        print(informationType)
        print('----------{}----------'.format('informativeness'))
        print(informativeness)
        print('----------{}----------'.format('sharingOwner'))
        print(sharingOwner)
        print('----------{}----------'.format('sharingOthers'))
        print(sharingOthers)

        ## privacy time in image wise
        image_privacy_time = {'Prolific': {}, 'CrowdWorks': {}, 'All': {}}
        privacy_time = {'Prolific': {0: 0, 1: 0, 2: 0}, 
                        'CrowdWorks': {0: 0, 1: 0, 2: 0}, 
                        'All': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}}
        for image_name in self.img_annotation_map.keys():
            all = 0
            for platform, annotations in self.img_annotation_map[image_name].items():
                this_platform = 0
                for i, annotation in enumerate(annotations):
                    if strict_mode and i >= strict_num:
                        break
                    if annotation not in image_privacy_time[platform].keys():
                        image_privacy_time[platform][annotation] = 0
                    if annotation not in image_privacy_time['All'].keys():
                        image_privacy_time['All'][annotation] = 0
                    with open(os.path.join(self.annotation_path, platform, 'labels', annotation), encoding="utf-8") as f_label:
                        ifPrivacy = False
                        label = json.load(f_label)
                        if len(label['manualAnnotation']) > 0:
                            ifPrivacy = True
                        for key, value in label['defaultAnnotation'].items():
                            category = value['category']
                            if ignore_prev_manual_anns and category.startswith('Object'):
                                continue
                            if not value['ifNoPrivacy']:
                                ifPrivacy = True
                        if ifPrivacy:
                            all += 1
                            this_platform += 1
                image_privacy_time[platform][annotation] = this_platform
                privacy_time[platform][this_platform] += 1

            image_privacy_time['All'][annotation] = all
            privacy_time['All'][all] += 1
        
        print('privacy', privacy_time)
    
        ### contigency table for information type, sharing owner, sharing others
        # information type
        # table 6*6
        information_tab = np.zeros((6, 6))
        for i, row in self.mega_table.iterrows():
            information = json.loads(row['informationType'])
            for j in range(6):
                if information[j] == 1:
                    information_tab[j] += information
        print(information_tab)

        # sharing owner
        # table 7*7
        # if a larger index == 1 while smaller index == 0, then it is unusual, except the case of index 0 and 6
        unusual = 0
        sharingOwner_tab = np.zeros((7, 7))
        for i, row in self.mega_table.iterrows():
            sharingOwner = json.loads(row['sharingOwner'])
            for j in range(7):
                if sharingOwner[j] == 1:
                    sharingOwner_tab[j] += sharingOwner
            # find unusual
            ifunusual = False
            for j in range(1, 5):
                for k in range(j + 1, 6):
                    if sharingOwner[j] == 0 and sharingOwner[k] == 1:
                        unusual += 1
                        ifunusual = True
                        break
                if ifunusual:
                    break
        print('unusual:', unusual)
        print(sharingOwner_tab)

        # sharing others
        # table 7*7
        unusual = 0
        sharingOthers_tab = np.zeros((7, 7))
        for i, row in self.mega_table.iterrows():
            sharingOthers = json.loads(row['sharingOthers'])
            for j in range(7):
                if sharingOthers[j] == 1:
                    sharingOthers_tab[j] += sharingOthers
            ifunusual = False
            for j in range(1, 5):
                for k in range(j + 1, 6):
                    if sharingOthers[j] == 0 and sharingOthers[k] == 1:
                        unusual += 1
                        ifunusual = True
                        break
                if ifunusual:
                    break
        print('unusual:', unusual)
        print(sharingOthers_tab)

    def bbox_iou(self,boxA, boxB):
    # boxA and boxB are expected to be lists or tuples of four numbers representing the (x, y, w, h) coordinates of the boxes

    # calculate the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # calculate the area of intersection rectangle
        intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # calculate the area of both bounding boxes
        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]

        # calculate the union area
        union_area = boxA_area + boxB_area - intersection_area

        # calculate IoU
        iou = intersection_area / union_area

        return iou
    
    def count_overlap_in_manual_table(self, split_count = False, count_scale = 'CrowdWorks')->None:
        # read mega table and select Manual label
        self.manual_table = pd.read_csv(self.mega_table_path)
        self.manual_table = self.manual_table[self.manual_table['category'] == 'Manual Label']
        #print len
        print('manual table len:', len(self.manual_table))
        # divide it by CrowdWorks and Prolific and out put length
        print('CrowdWorks len:', len(self.manual_table[self.manual_table['platform'] == 'CrowdWorks']))
        print('Prolific len:', len(self.manual_table[self.manual_table['platform'] == 'Prolific']))
        #print unique image path len
        if split_count:
            self.manual_table = self.manual_table[self.manual_table['platform'] == count_scale]
        print('unique image path len:', len(self.manual_table['imagePath'].unique()))
        #imagePath bounding box map
        imagePath = {}
        for i, row in self.manual_table.iterrows():
            if row['imagePath'] not in imagePath.keys():
                imagePath[row['imagePath']] = []
            bbox = json.loads(row['bbox'])
            bbox = bbox[0]
            imagePath[row['imagePath']].append(bbox)
        
        # count overlap in bounding box over a threshold
        threshold = 0.7
        overlap = 0
       
        #Non-Maximum Suppression if overlap > threshold
        filtered_imagePath = {}
        for key, value in imagePath.items():
            value = np.array(value)
            value = self.non_max_suppression(value, threshold)
            filtered_imagePath[key] = value

        #count number of filtered bounding box
        for key, value in filtered_imagePath.items():
            overlap += len(value)
        print('overlap:', overlap)

        #count specific overlap by filtered bounding box
        overlap = {}
        for key, value in filtered_imagePath.items():
            for bbox in value:
                overlap_num = 0
                for ori_bbox in imagePath[key]:
                    if self.bbox_iou(bbox, ori_bbox) > threshold:
                        overlap_num += 1
                if overlap_num not in overlap.keys():
                    overlap[overlap_num] = 1
                else:
                    overlap[overlap_num] += 1
        print('overlap:', overlap)  

    def visualDistribution(self, visualization=False):
        #target: get the distribution of all visual content by different metrics, like size, foreground or background, and relative location to the center.
        #Each distribution has three or two categories, like small, medium, large, foreground, background, close, netural, far, etc.
        # read mega table 

        self.mega_table = pd.read_csv(self.mega_table_path)
        
        # for each item, get the image path, and get the image
        # get the bounding box from mega table, and get the size of the bounding box and size of the image through PIL.image
        # get the relative location of the bounding box to the center of the image
        # read every row of the meage table
        relative_sizes = []
        relative_position = []
        width_height_ratio = []
        bbox_name=[]
        bboxes = []
        informativeness = []
        information_type = []
        # check if mega_table has category named manual label 
        print(self.mega_table)
        for index, row in self.mega_table.iterrows():
            #bboxes = json.loads(row['bbox'])
            image_path = os.path.join('images', row['imagePath'])
            image = Image.open(image_path)
            image_width, image_height = image.size
            for bbox in row['bbox']:
                #size
                size = bbox[2] * bbox[3]
                relative_size = size / (image_width * image_height)
                relative_sizes.append(relative_size)
                #relative location
                center_x = image_width / 2
                center_y = image_height / 2
                relative_x = (bbox[0] + bbox[2] / 2 - center_x) / image_width
                relative_y = (bbox[1] + bbox[3] / 2 - center_y) / image_height
                #print(relative_x, relative_y)
                relative_position.append([relative_x, relative_y])
                #width heigh ratio
                width_height_ratio.append(bbox[2] / bbox[3])
                bbox_name.append(row['imagePath'] + '_' + row['category'])
                informativeness.append(row['informativeness'])
                information_type.append(row['informationType'][:-1])
                bboxes.append(bbox)
        #visualize the distribution in coordinate system
        relative_sizes = np.array(relative_sizes)
        relative_position = np.array(relative_position)
        width_height_ratio = np.array(width_height_ratio)
        informativeness = np.array(informativeness)
        information_type = np.array(information_type)
        bboxes = np.array(bboxes)
        bbox_name = np.array(bbox_name)

        # divide data into 30, 40 ,30
        low_number = 30
        high_number = 70

        # Divide data into 30, 40 ,30
        low_size, high_size = np.percentile(relative_sizes, [low_number, high_number])
        low_ratio, high_ratio = np.percentile(width_height_ratio, [low_number, high_number])

        lowest_30_size = relative_sizes[relative_sizes < low_size]
        middle_40_size = relative_sizes[(relative_sizes >= low_size) & (relative_sizes <= high_size)]
        highest_30_size = relative_sizes[relative_sizes > high_size]

        lowest_30_ratio = width_height_ratio[width_height_ratio < low_ratio]
        middle_40_ratio = width_height_ratio[(width_height_ratio >= low_ratio) & (width_height_ratio <= high_ratio)]
        highest_30_ratio = width_height_ratio[width_height_ratio > high_ratio]
        # Compute the Euclidean distance of each point from the origin
        # Compute the Euclidean distance of each point from the origin
        distances = np.sqrt(np.sum(np.square(relative_position), axis=1))

        # Divide the distances into quantiles
        low_distance, high_distance = np.percentile(distances, [low_number, high_number])

        # Divide the positions based on these distances
        lowest_30_position = relative_position[distances < low_distance]
        middle_40_position = relative_position[(distances >= low_distance) & (distances <= high_distance)]
        highest_30_position = relative_position[distances > high_distance]

        # Compute the median distance for each group
        lowest_30_distance = np.median(distances[distances < low_distance])
        middle_40_distance = np.median(distances[(distances >= low_distance) & (distances <= high_distance)])
        highest_30_distance = np.median(distances[distances > high_distance])

        # Print the median distance for each group
        print('Median distance for the closest '+str(low_number)+'% of points:', lowest_30_distance)
        print('Median distance for the middle '+str(high_number-low_number)+'% of points:', middle_40_distance)
        print('Median distance for the farthest '+str(100-high_number)+'% of points:', highest_30_distance)
        print('max distance and min distance:', np.max(distances), np.min(distances))

        print('Median size of the smallest '+str(low_number)+'% of boxes:', np.median(lowest_30_size))
        print('Median size of the middle '+str(high_number-low_number)+'% of boxes:', np.median(middle_40_size))
        print('Median size of the largest '+str(100-high_number)+'% of boxes:', np.median(highest_30_size))
        print('max size and min size:', np.max(relative_sizes), np.min(relative_sizes))

        print('Median ratio of the smallest '+str(low_number)+'% of boxes:', np.median(lowest_30_ratio))
        print('Median ratio of the middle '+str(high_number-low_number)+'% of boxes:', np.median(middle_40_ratio))
        print('Median ratio of the largest '+str(100-high_number)+'% of boxes:', np.median(highest_30_ratio))
        print('max ratio and min ratio:', np.max(width_height_ratio), np.min(width_height_ratio))

        # print the average and std for each group 
        # distance
        print('Average distance for the closest '+str(low_number)+'% of points:', np.mean(distances[distances < low_distance]))
        print('Average distance for the middle '+str(high_number-low_number)+'% of points:', np.mean(distances[(distances >= low_distance) & (distances <= high_distance)]))
        print('Average distance for the farthest '+str(100-high_number)+'% of points:', np.mean(distances[distances > high_distance]))
        print('std distance for the closest '+str(low_number)+'% of points:', np.std(distances[distances < low_distance]))
        print('std distance for the middle '+str(high_number-low_number)+'% of points:', np.std(distances[(distances >= low_distance) & (distances <= high_distance)]))
        print('std distance for the farthest '+str(100-high_number)+'% of points:', np.std(distances[distances > high_distance]))
        # size
        print('Average size of the smallest '+str(low_number)+'% of boxes:', np.mean(lowest_30_size))
        print('Average size of the middle '+str(high_number-low_number)+'% of boxes:', np.mean(middle_40_size))
        print('Average size of the largest '+str(100-high_number)+'% of boxes:', np.mean(highest_30_size))
        print('std size of the smallest '+str(low_number)+'% of boxes:', np.std(lowest_30_size))
        print('std size of the middle '+str(high_number-low_number)+'% of boxes:', np.std(middle_40_size))
        print('std size of the largest '+str(100-high_number)+'% of boxes:', np.std(highest_30_size))

        # ratio
        print('Average ratio of the smallest '+str(low_number)+'% of boxes:', np.mean(lowest_30_ratio))
        print('Average ratio of the middle '+str(high_number-low_number)+'% of boxes:', np.mean(middle_40_ratio))
        print('Average ratio of the largest '+str(100-high_number)+'% of boxes:', np.mean(highest_30_ratio))
        print('std ratio of the smallest '+str(low_number)+'% of boxes:', np.std(lowest_30_ratio))
        print('std ratio of the middle '+str(high_number-low_number)+'% of boxes:', np.std(middle_40_ratio))
        print('std ratio of the largest '+str(100-high_number)+'% of boxes:', np.std(highest_30_ratio))

        
        
        df = pd.DataFrame(columns=['image_path', 'category', 'bbox', 'informationType', 'informativeness', 'relative_size', 'relative_position', 'width_height_ratio'])
        for i in range(len(relative_sizes)):
            df.loc[i] = [bbox_name[i].split('_')[0], bbox_name[i].split('_')[1], bboxes[i], information_type[i], informativeness[i], relative_sizes[i], relative_position[i], width_height_ratio[i]]
        # Add a new column to identify the group of each row
        df['size_group'] = pd.cut(df['relative_size'], bins=[-np.inf, low_size, high_size, np.inf], labels=['low', 'middle', 'high'])
        df['position_group'] = pd.cut(df['relative_position'].apply(lambda x: np.sqrt(x[0]**2 + x[1]**2)), bins=[-np.inf, low_distance, high_distance, np.inf], labels=['low', 'middle', 'high'])
        df['ratio_group'] = pd.cut(df['width_height_ratio'], bins=[-np.inf, low_ratio, high_ratio, np.inf], labels=['low', 'middle', 'high'])
        
        # Group by image_path, category, and the new group columns, then aggregate the bboxes and informativeness
        grouped = df.groupby(['image_path', 'category', 'size_group', 'position_group', 'ratio_group']).agg({
            'bbox': lambda x: [list(b) for b in x],  # convert each bbox into a list, resulting in a list of lists
            'informativeness': lambda x: np.round(np.mean(x), 2), # calculate the mean informativeness, to 2 decimal places
            'informationType': lambda x: np.any(np.array(list(x)), axis=0).astype(int).tolist() # calculate the logical OR of the informationType, to get a list of 0s and 1s
        }).reset_index()
        #print unique bbox
        #remove all column if the informativeness or bbox is nan
        grouped = grouped.dropna(subset=['informativeness', 'bbox'])
        # save the dataframe
        # randomly choose sample_size rows
        # Stratified sampling
        grouped['informativeness_group'] = pd.cut(grouped['informativeness'], bins=[-np.inf, 2, 4.0001, np.inf], labels=['low', 'middle', 'high'])
        grouped['group (size, position, ratio)'] = grouped['size_group'].astype(str) + "_" + grouped['position_group'].astype(str) + "_" + grouped['ratio_group'].astype(str)
        
        # print number of each group combination
        print(grouped['group (size, position, ratio)'].value_counts())

        if visualization:
            plt.scatter(relative_position[:, 0], relative_position[:, 1], s=relative_sizes * 1000, c=width_height_ratio, cmap='viridis')
            plt.colorbar()
            #add titile
            plt.title('relative position and size of bounding box')
            plt.show()
            #visualize the distribution in histogram
            plt.hist(relative_sizes, bins=20)
            plt.title('relative size of bounding box')
            plt.show()
            plt.hist(relative_position[:, 0], bins=20)
            plt.title('relative x position of bounding box')
            plt.show()
            plt.hist(relative_position[:, 1], bins=20)
            plt.title('relative y position of bounding box')
            plt.show()
            plt.hist(width_height_ratio, bins=20)
            plt.title('width height ratio of bounding box')
            plt.show()

    def prepare_json_for_study1(self):
        # create one file that scores all privacy-threatening objects for each image
        # we exclude the manual annotations for simplicity
        # the category will be 22 categories + others
        self.mega_table = pd.read_csv(self.mega_table_path)
        
        # read all annotations in dataset for each image, not divided into platform
        annotations = {}
        for platform in ['CrowdWorks', 'Prolific']:
            files = os.listdir(os.path.join(self.annotation_path, platform, 'labels'))
            for file in files:
                with open(os.path.join(self.annotation_path, platform, 'labels', file), encoding="utf-8") as f:
                    annotation = json.load(f)
                    image_name = file.split('_')[0]
                    if image_name not in annotations.keys():
                        annotations[image_name] = {}
                    for key, value in annotation['defaultAnnotation'].items():
                        if value['ifNoPrivacy']:
                            continue
                        category = value['category']
                        if category not in annotations[image_name].keys():
                            annotations[image_name][category] = []
                        annotations[image_name][category].append(value['bbox'])

if __name__ == '__main__':
    analyze = analyzer()

    analyze.distribution()
    analyze.basic_info()
    analyze.basic_count(ignore_prev_manual_anns=False,split_count=False,count_scale='Prolific')
    analyze.count_frequency()
    analyze.count_overlap_in_manual_table(split_count=False, count_scale='Prolific')
    analyze.visualDistribution(visualization=True)