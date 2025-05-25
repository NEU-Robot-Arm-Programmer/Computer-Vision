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

