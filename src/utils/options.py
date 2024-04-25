import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="Basemodel")
    parser.add_argument("--algorithm", type=str, default="Fedmem",
                        choices=["pFedme", "FedAvg", "Fedmem", "FeSEM", "FedProx"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.7,
                        help="Regularizer for pFedme")
    parser.add_argument("--K", type=int, default=30,
                        help="Regularizer for pFedme")

    parser.add_argument("--user_ids", type=list, default=[['71', '7', '189', '202', '208', '0', '160', '10', '105', '68', '139', '207', '57', 
                                                         '128', '133', '190', '149', '290', '117', '253', '67', '76', '145', '162', '49', '17', 
                                                         '279', '200', '194', '3', '51', '204', '21', '22', '94', '206', '79', '30', '129', '115', 
                                                         '225', '184', '281', '88', '58', '295', '161', '250', '50', '63', '176', '275', '52', 
                                                         '35', '109', '240', '219', '138', '291', '62', '224', '267', '98', '252', '178', '106', '146', 
                                                         '169', '80', '75', '60', '44', '180', '116', '210', '144', '203', '37', '229', '212', '196', 
                                                         '33', '259', '187', '235', '300', '99', '181', '273', '172', '173', '218', '110', '248', '143', '228', 
                                                         '107', '92', '270', '175', '19', '286', '89', '277', '155', '55', '97', '93', '74', '182', '205', '157', 
                                                         '234', '39', '124', '28', '40', '73', '217', '126', '122', '8', '16', '54', '4', '14', '215', '121', '241', 
                                                         '245', '112', '61', '233', '256', '83', '168', '118', '64', '20', '177', '268', '147', '274', '158',
                                                         '195', '287', '77', '265', '211', '53', '81', '86', '134', '123', '154', '127', '216', '188', '56', '137', 
                                                         '103', '41', '2', '232', '26', '18', '1', '131', '288', '239', '31', '223', '231', '43', '263', '151', 
                                                         '140', '132', '159', '119', '91', '130', '6', '101', '255', '237', '25', '152', '87', 
                                                         '32', '227', '243', '111', '294', '42', '257', '183', '27', '90', '299', '214', '222', '298', '284',
                                                         '66', '251', '125', '269', '164', '271', '45', '198', '84', '246', '29', '150', '238', '293', '264', '156',
                                                         '12', '272', '70', '266', '260', '249', '199', '142', '258', '48', '69', '114', '15', '85', '283', 
                                                         '254', '135', '59', '197', '171', '297', '247', '120', '292', '179', '191', '113', '100', '95', '65', '46', 
                                                         '230', '136', '192', '236', '34', '186', '170', '36', '102', '276', '24', '221', '23', '11', '167', 
                                                         '153', '193', '13', '104', '201', '185', '261', '148', '9', '163', '78', '96', '209', '47'],
                                                        ['329', '306', '325', '504', '584', '323', '369', '470', '597', '587', '524', '488', '482', '442', '503', 
                                                         '408', '406', '544', '515', '432', '491', '327', '425', '324', '590', '366', '307', '346', '486', '435', 
                                                         '446', '413', '576', '500', '391', '433', '538', '460', '509', '326', '492', '478', '551', '393', 
                                                         '573', '342', '417', '469', '351', '461', '502', '392', '512', '511', '497', '533', '537', '335', '588', 
                                                         '526', '563', '394', '494', '571', '558', '472', '523', '554', '395', '349', '310', '400', '570', '313', 
                                                         '361', '427', '371', '303', '519', '383', '431', '315', '308', '378', '385', '443', '539', '549', '479', 
                                                         '490', '328', '455', '487', '465', '334', '550', '579', '396', '305', '568', '501', '447', '569', '564', 
                                                         '473', '380', '330', '561', '436', '403', '354', '388', '372', '418', '320', '529', '532', '459', '596', 
                                                         '404', '543', '440', '517', '449', '438', '430', '441', '565', '522', '332', '542', '581', '448', '386', 
                                                         '589', '412', '364', '437', '347', '559', '466', '574', '471', '575', '489', '390', '445', '401', '339', 
                                                         '304', '363', '426', '370', '540', '398', '592', '531', '387', '547', '506', '483', '484', '505', '423', 
                                                         '373', '508', '428', '381', '481', '578', '536', '514', '527', '451', '555', '345', '462', '316', '374', 
                                                         '485', '552', '376', '595', '384', '365', '562', '405', '407', '513', '582', '311', '410', '356', '318', 
                                                         '530', '414', '434', '545', '424', '343', '336', '566', '535', '420', '599', '586', '416', '358', '463', 
                                                         '567', '340', '439', '507', '591', '464', '453', '409', '312', '560', '422', '577', '516', '580', '368', 
                                                         '333', '454', '309', '553', '415', '322', '357', '375', '521', '572', '528', '302', '362', '510', '477', 
                                                         '520', '355', '399', '411', '317', '456', '352', '389', '541', '402', '344', '350', '583', '452', '444', 
                                                         '321', '419', '337', '474', '467', '476', '548', '359', '499', '546', '301', '367', '421', '379', '314', 
                                                         '556', '495', '496', '480'],
                                                         ['71', '7', '189', '202', '208', '0', '160', '10', '105', '68', '139', '207', '57', 
                                                         '128', '133', '190', '149', '290', '117', '253', '67', '76', '145', '162', '49', '17', 
                                                         '279', '200', '194', '3', '51', '204', '21', '22', '94', '206', '79', '30', '129', '115', 
                                                         '225', '184', '281', '88', '58', '295', '161', '250', '50', '63', '176', '275', '52']])
    parser.add_argument("--country", type=str, default="japan", choices=["japan", "uk"])
    
    parser.add_argument("--lambda_1", type=float, default=0.25, 
                        help="Regularization term lambda_1")
    parser.add_argument("--lambda_2", type=float, default=0.25, 
                        help="Regularization term lambda_2")
    parser.add_argument("--gamma", type=float, default=0.05, 
                        help="regularization term gamma for PerMFL and scale parameter for RBF kernel in Fedmem")
    parser.add_argument("--alpha", type=float, default=0.05, 
                        help="learning rate for local models in fedmem")
    parser.add_argument("--eta", type=float, default=0.01, 
                        help="personalization parameter for Fedmem")
    
    parser.add_argument("--num_global_iters", type=int, default=5)
    parser.add_argument("--local_iters", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="SGD")
    
    parser.add_argument("--times", type=int, default=1, 
                        help="running time")
    parser.add_argument("--exp_start", type=int, default=0,
                        help="experiment start no")
    parser.add_argument("--gpu", type=int, default=0,
                        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    
    parser.add_argument("--users_frac", type=float, default=1.0, 
                        help="selected fraction of users available per global round")
    parser.add_argument("--total_users", type=int, default=40, 
                        help="total participants")
    parser.add_argument("--data_silo", type=int, default=100)
   
    parser.add_argument("--num_teams", type=int, default=5,
                        help="Number of teams")
    parser.add_argument("--p_teams", type=int, default=1,
                        help="number of team selected per global round")
    parser.add_argument("--cluster", type = str, default="dynamic", choices=["apriori_hsgd", "dynamic", "apriori"])
    parser.add_argument("--target", type=int, default=10, choices=[3,10], help="number of target classes")

    parser.add_argument("--fixed_user_id", type=int, default=16)
    parser.add_argument("--fix_client_every_GR", type=int, default=0, choices=[0,1])

    args = parser.parse_args()

    return args
