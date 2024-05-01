        num_rows = len(self.mega_table)
        train_per, val_per, test_per = 65, 10, 25
        train_size = int((train_per/100.0) * num_rows)
        val_size = int((val_per/100.0) * num_rows)
        test_size = num_rows - train_size - val_size

        train_df = self.mega_table.sample(n=train_size, random_state=0)
        rem_df = self.mega_table.drop(train_df.index)
        val_df = rem_df.sample(n=val_size, random_state=0)
        test_df = rem_df.drop(val_df.index)

        dataset_files_dir = "dataset_files/%s/" % self.algorithm
        if os.path.exists(dataset_files_dir):
            shutil.rmtree(dataset_files_dir)
        os.makedirs(dataset_files_dir)

        train_df.to_csv("%s/train_%d.csv" % (dataset_files_dir, self.id), index=False)
        val_df.to_csv("%s/val_%d.csv" % (dataset_files_dir, self.id), index=False)
        test_df.to_csv("%s/test_%d.csv" % (dataset_files_dir, self.id), index=False)

        train_dataset = ImageMaskDataset(train_df, image_folder, input_channel, image_size, flip = True)
        val_dataset = ImageMaskDataset(val_df, image_folder, input_channel, image_size)
        test_dataset = ImageMaskDataset(test_df, image_folder, input_channel, image_size)
