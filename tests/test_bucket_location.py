from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket("gs://llm_pruning_us_central2_b ")
print(bucket.location)