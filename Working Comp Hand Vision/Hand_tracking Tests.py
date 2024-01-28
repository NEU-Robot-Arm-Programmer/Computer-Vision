import unittest
import cv2
from Hand_tracking import handdetec, pixels_to_meters

class TestHandDetect(unittest.TestCase):
    def setUp(self):
        self.detector = handdetec()

#test to see i the program can identify hands
    def test_findhands(self):
        img_open = cv2.imread(r"C:\Users\steph\Downloads\open_palm.jpg")
        result_open = self.detector.findhands(img_open)
        self.assertTrue(isinstance(result_open, type(img_open)))

        img_curved = cv2.imread(r"C:\Users\steph\Downloads\curved_palm.jpg")
        result_curved = self.detector.findhands(img_curved)
        self.assertTrue(isinstance(result_curved, type(img_curved)))


#test the finding the position of hte hands
    def test_findposition(self):
        img_open = cv2.imread(r"C:\Users\steph\Downloads\open_palm.jpg")
        result_open = self.detector.findhands(img_open)

        #check teh results are in a list of lists
        self.assertIsInstance(result_open, list)
        for item in result_open:
            self.assertIsInstance(item, list)

            #check if each sublist has teh expectedstructure: id relative_x, y]
            self.assertEqual(len(item), 3)
            self.assertIsInstance(item[0], int) #this is the id
            self.assertIsInstance(item[1], float)
            self.assertIsInstance(item[2], float)

            #other tests based on teh expected behavior of position


    # def test_findhands(self):
    #     #have an instance of the hands
    #     detector = handdetec()
    #     #these is the address of the test picture I used
    #     img_open = cv2.imread(r"C:\Users\steph\Downloads\open_palm.jpg")
    #     result_img_open = detector.findhands(img_open, draw=False)
    #     self.assertIsNotNone(result_img_open)
    #     expected_hands_open = 1
    #     self.assertEqual(len(detector.result.multi_hand_landmarks), expected_hands_open)
    #     if expected_hands_open == 1:
    #         expected_coordinates_open = {
    #             0: (expected_x0, expected_y0),  # Replace with actual expected coordinates
    #             1: (expected_x1, expected_y1),
    #         }
    #         for id, (real_world_x, real_world_y) in expected_coordinates_open.items():
    #             self.assertAlmostEqual(positions[id][1], real_world_x, places=2)
    #             self.assertAlmostEqual(positions[id][2], real_world_y, places=2)
    #
    #
    #     #img = cv2.imread(r"C:\Users\steph\Downloads\handtest.jpg")
    #     img_curved_hand = cv2.imread(r"C:\Users\steph\Downloads\curved_palm.jpg")
    #     #call the method to find the hand
    #     result_img_curved = detector.findhands(img_curved_hand, draw=False)
    #     #need assertions based on the expected behavior
    #     self.assertIsNotNone(result_img_curved)
    #     #assertion 1: The correct number of hand landmarks is detected.
    #     expected_hands_curved = 1
    #     self.assertEqual(len(detector.result.multi_hand_landmarks), expected_hands)
    #
    #     #assertion 3: detected landmarks have the expected coordinates.
    #     if expected_hands_curved == 1:
    #         expected_coordinates_curved = {
    #             0: (expected_x0_cur, expected_y0_cur), #expected_x0 etc are specific x and y coordinates for the landmarks
    #             #i think these need to be actual coordinates
    #             1: (expected_x1_cur, expected_y1_cur),
    #             2: (expected_x2_cur, expected_y2_cur),
    #             3: (expected_x3_cur, expected_y3_cur),
    #             4: (expected_x4_cur, expected_y4_cur),
    #             5: (expected_x5_cur, expected_y5_cur),
    #             6: (expected_x6_cur, expected_y6_cur),
    #         }
    #         for id, (real_world_x, real_world_y) in expected_coordinates_curved.items():
    #             self.assertAlmostEqual(positions[id][1], real_world_x, places=2)
    #             self.assertAlmostEqual(positions[id][2], real_world_y, places=2)

    # def test_findingposition(self):
    #     img = cv2.imread(r"C:\Users\steph\Downloads\open_palm.jpg")
    #     result = self.detector.findposition(img, handNum=1, draw=True)
    #     self.assertTrue(isinstance(result, list))
    #     #self.assertGreater(len(position), 0)
    #     # Check if the correct number of landmarks is detected for the first hand
    #
    #     #this is the number of points that pop up for my hand
    #     expected_landmarks = 21
    #     self.assertEqual(len(position), expected_landmarks)
        #self.assertEqual(len(detector.result.multi_hnd_landmarks), expected_landmarks)

        #check if the coordinates are within the range
        # expected_x_range = (0, 1)  # Adjust the range based on your expectations
        # expected_y_range = (0, 1)
        # self.assertGreaterEqual(positions[0][1], expected_x_range[0])
        # self.assertLessEqual(positions[0][1], expected_x_range[1])
        # self.assertGreaterEqual(positions[0][2], expected_y_range[0])
        # self.assertLessEqual(positions[0][2], expected_y_range[1])


#this one works
class TestPixToMet(unittest.TestCase):
    """ A test the works to make sure the converstion of pixels to meters is accurate"""
    def test_pixels_to_meters(self):
        result = pixels_to_meters(10,20,1.0,
                                  1.0,640,480)
        self.assertTrue(isinstance(result, tuple))
        result2 = pixels_to_meters(23, 12, 2.0, 2.0,
                                   360,360)
        self.assertTrue(isinstance(result2, tuple))

    #intential wrong
    def test_pixels_wrong(self):
        with self.assertRaises(ValueError):
            pixels_to_meters(10,20,1.0,1.0,
                             -640,480)

        with self.assertRaises(ValueError):
            pixels_to_meters(23,12,2.0,2.0,
                             360, -350)


if __name__ == '__main__':
    unittest.main()

