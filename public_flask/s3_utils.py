import os
import boto3


class S3Utils:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(S3Utils, cls).__new__(cls)
            cls._instance.client = boto3.client('s3')
        return cls._instance

    def upload_directory_to_s3(self, local_directory, bucket_name, s3_directory):
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_file = os.path.join(root, file)
                s3_file = os.path.join(
                    s3_directory, os.path.relpath(local_file, local_directory))
                self.client.upload_file(local_file, bucket_name, s3_file)

    def delete_files_in_s3_directory(self, bucket_name, s3_directory):
        # Fetch the list of objects in the specified S3 directory
        objects_to_delete = self.client.list_objects(
            Bucket=bucket_name, Prefix=s3_directory)

        # Delete the objects
        for obj in objects_to_delete.get('Contents', []):
            self.client.delete_object(Bucket=bucket_name, Key=obj['Key'])
