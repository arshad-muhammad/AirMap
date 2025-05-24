import ee

# Authenticate (only needed once)
ee.Authenticate()

# Initialize with your project
ee.Initialize(project='feedscan')

# Example operation
print(ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3').size().getInfo())
