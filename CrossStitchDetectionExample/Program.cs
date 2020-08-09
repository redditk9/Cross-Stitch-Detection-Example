/* This example was written by Reddit user redditk9. In response to the post:
 * https://www.reddit.com/r/csharp/comments/i52apl/where_to_start_with_image_analysis_symbol/
 * 
 * This particular example shows how to detect all locations of ONE of the symbols and follows the same process
 * described in a comment by myself. The process can simply be put into a function and repeated for each symbol.
 * 
 * Unforuntely I have to put this code out under the GNU GPL License V3 since it makes use of EmguCV which is
 * only available in the unpaid version as GNU GPL License V3. For license information, see LICENSE.txt.
 * 
 * I apologize for the excessive comments. It's meant to be educational.
 * */

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CrossStitchDetectionExample
{
    class Program
    {
        static void Main(string[] args)
        {
            using(DisposableList garbage = new DisposableList())
            {
                // ===== LOADING IMAGES. ====== //
                // Note: Images are immediately added to the garbage. This ensures it is disposed if a failure occurs.
                //     In this toy example it doesn't matter much, but consider this in real world situations.

                Image<Gray, byte> cvOriginal = new Image<Gray, byte>("./Images/Original.jpg"); // Original image.
                garbage.Add(cvOriginal);

                // You can choose whichever symbol you like. The available ones are:
                // Shape_Box
                // Shape_C
                // Shape_Down
                // Shape_Heart
                // Shape_L
                // Shape_Left
                // Shape_U
                Image<Gray, byte> cvTemplate = new Image<Gray, byte>("./Images/Shape_Box.jpg"); // Symbol template.
                garbage.Add(cvTemplate);


                // ===== 1. THRESHOLD IMAGE ===== //
                // This step converts the original image into black and white to remove a lot of the noise in the image.
                // This will only work well if the original image is uniform in lighting, which in this case is true.
                // If you have a situation where lighting is not uniform, consider Adaptive Thresholding or CLAHE.

                // The pixel value to seperate between black and white. This can be found by just playing around experimentally.
                // More advanced techniques can do this automatically like K-Means or simply thresholding at the mean value.
                int thresholdLevel = 128; 

                // Threshold image. It is inverted ("Inv") here so the symbols are white with a black background.
                Image<Gray, byte> cvThreshold = cvOriginal.ThresholdBinaryInv(new Gray(thresholdLevel), new Gray(255));
                garbage.Add(cvThreshold);

                CvInvoke.Imshow("1. Thresholded Image", cvThreshold); // Imshow displays a window with the image.


                // ===== 2. TEMPLATE MATCHING ===== //
                // This technique uses normalized cross-correlation to detect potential match locations.
                // "In essence" the algorithm slides the template over the original image and returns a value
                // for each pixel with the quality ("correlation coefficient") of the match. 1.0 is a perfect match and -1.0 is a perfect inverse match.
                // The spots with the highest correlation coefficients are the most likely match positions. 
                // In the displayed image, you will see bright white dots where the likely matches are and gray regions where there are similar matches.

                Image<Gray, float> cvNormXCorrelation = cvThreshold.MatchTemplate(cvTemplate, TemplateMatchingType.CcoeffNormed);
                garbage.Add(cvNormXCorrelation);

                CvInvoke.Imshow("2. Cross Correlation", cvNormXCorrelation);


                // ===== 3. THRESHOLDING ROUND 2 ===== //
                // Now that we have the likely locations, we want to single out areas with high correlation coefficients.
                // Very simply put, we only look at areas where the correlation coefficient is above a certain number.

                // The higher the value, the more sensitive. Finding the correct value to use here can either be done experimentally, or
                // statistically in real-time applications.
                float minimumCorrelationCoefficient = 0.80f;

                // The image gets thresholded as a float values (0.0f or 1.0f). We will prefer to work with byte values (0 or 255) so we convert.
                Image<Gray, float> cvDetectionsFloat = cvNormXCorrelation.ThresholdBinary(new Gray(minimumCorrelationCoefficient), new Gray(255.0f));
                cvDetectionsFloat._Mul(255); // Multiply all 1.0f's by 255.

                Image<Gray, byte> cvDetections = cvDetectionsFloat.Convert<Gray, byte>(); // Convert to bytes.
                garbage.Add(cvDetections);

                cvDetectionsFloat.Dispose(); // We can dispose this here. It was only needed for a short period of time.

                CvInvoke.Imshow("3. Detections", cvDetections);


                // ===== 4. BINARY LARGE OBJECTS DETECTION (BLOB) ====== //
                // This step is a bit excessive since the matches in this example are near perfect. The valid locations are only several pixels wide.
                // However, in real life with more complicated images, the detected locations could be larger. Additionally, you may have small pockets
                // of detected areas which are not the true area. Therefore, you might be interested in finding the center of the detected areas and 
                // filtering them. Additionally, this step will convert from an image of valid locations into a list of points which is the end goal.

                // I am not using almost any of filtering here because the detected areas are near perfect, but you can use these for other situations.
                // This is a very powerful tool. I highly recommend looking into these for other applications.
                SimpleBlobDetectorParams blobDetectorParams = new SimpleBlobDetectorParams()
                {
                    FilterByArea = false,
                    FilterByCircularity = false,
                    FilterByColor = false,
                    FilterByConvexity = false,
                    FilterByInertia = false,
                    MinArea = 0.0f,
                    MaxArea = float.PositiveInfinity,


                    // This is the only filter I use since we KNOW the symbols are on a grid with a specific spacing.
                    // If by some chance there are multiple detections in a small area, it will pick one.
                    // The spacing is a bit less than the actual spacing.
                    MinDistBetweenBlobs = 50 
                };

                // Perform blob detection.
                SimpleBlobDetector blobDetector = new SimpleBlobDetector(blobDetectorParams);
                MKeyPoint[] detectedLocations = blobDetector.Detect(cvDetections); // MKeyPoint is just a location with some additional information.

                // The detected locations are the TOP-LEFT of the match. If we want the points to be in the center, we can just add by half the template size to x and y.
                float offsetX = cvTemplate.Width / 2.0f;
                float offsetY = cvTemplate.Height / 2.0f;
                IEnumerable<PointF> finalLocations = detectedLocations.ToList().Select(mkp => new PointF(mkp.Point.X + offsetX, mkp.Point.Y + offsetY));


                // ===== Here we draw the BLOB detection results. ===== //
                Image<Bgr, byte> cvResult = cvOriginal.Convert<Bgr, byte>(); // Make the original image a color image so we can draw with nice colors.
                garbage.Add(cvResult);

                // Draw each detected point.
                foreach(PointF location in finalLocations)
                {
                    cvResult.Draw(new Cross2DF(location, 15, 15), new Bgr(Color.Magenta), 2);
                }

                CvInvoke.Imshow("4. Result from Blob Detection", cvResult);


                CvInvoke.WaitKey(0); // Wait for user to press a button.
                CvInvoke.DestroyAllWindows(); // Close all windows.

                // Done!
            }
        }
    }

    /// <summary>
    /// This class is simply a disposable list. 
    /// Image processing tends to generate lots of IDisposable's since the images are disposable.
    /// It is useful to have a simple class like this so that the 'using' statement can be easily used.
    /// </summary>
    class DisposableList : List<IDisposable>, IDisposable
    {
        public void Dispose()
        {
            this.ForEach(i => i.Dispose());
        }
    }
}
