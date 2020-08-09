Cross Stitch Detection Example
===

Written by: Reddit User **u/redditk9**

This example was written in response to the post:
"[Where to start with image analysis symbol](https://www.reddit.com/r/csharp/comments/i52apl/where_to_start_with_image_analysis_symbol/)" written by **u/format71**.
 
This example demonstrates how symbols locations could be detected in an image of cross stitch patterns. Image processing chains are always specific to the problem. This example is no exception. This solution makes the following assumptions:

1. The image is aligned i.e. it is not rotated at all.
2. The image has only very small distortions.
3. The image has uniform lighting.

Some work was done beforehand with ImageJ to:
1.  Align and crop the original image.
2.  Crop each of the symbol images and threshold them.

How It Works (Overview)
---
Step 1. The original image is thresholded to remove noise.  

```csharp
Image<Gray, byte> cvThreshold = cvOriginal.ThresholdBinaryInv(new Gray(thresholdLevel), new Gray(255));
```

Step 2. Normalized cross correlation is used to detect likely match locations.  

```csharp
Image<Gray, float> cvNormXCorrelation = cvThreshold.MatchTemplate(cvTemplate, TemplateMatchingType.CcoeffNormed);
```

Step 3. The cross correlation result is thresholded to create a binary image of high probability locations.  

```csharp
Image<Gray, float> cvDetectionsFloat = cvNormXCorrelation.ThresholdBinary(new Gray(minimumCorrelationCoefficient), new Gray(255.0f));
```

Step 4. A BLOB detector is used to find the centers of each high probability location.  

```csharp
SimpleBlobDetector blobDetector = new SimpleBlobDetector(blobDetectorParams);
MKeyPoint[] detectedLocations = blobDetector.Detect(cvDetections);
```

Done!
