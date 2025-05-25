import unittest
import cv2
#from Hand_tracking import handdetec, pixels_to_meters
from main import handdetec, pixels_to_meters

#Test 1. Make sure the camera opens correctly and that the program opens correctly
#Test 2. make sure that fund hands is able to actually see and object and recognize them as hands
#Test 3. test positions are done correctly
#Test 4. Test teh conversion of pixels to meters is correct (use unitconversion.org
#Test 5. Test that calculate angles is done correctly

class TestHandDetect(unittest.TestCase):
    def setUp(self):
        self.detector = handdetec()

#test to see i the program can identify hands
    def test_findhands(self):
        """Test if teh find hands is able to detect actual hands(it should) and return teh correct statement"""
        img_open = cv2.imread("open_palm.jpg")
        result_open = self.detector.findhands(img_open)
        #make sure that find_hands returns angles
        self.assertIsInstance(result_open, dict)
        #self.assertTrue(isinstance(result_open, type(img_open)))
        #do the same with curved
        img_curved = cv2.imread("closed_hand.jpg")
        #img_curved = cv2.imread(r"C:\Users\steph\Downloads\curved_palm.jpg")
        result_curved = self.detector.findhands(img_curved)
        self.assertIsInstance(result_curved, dict)
        # type(img_curved)))


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



#this one works
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

