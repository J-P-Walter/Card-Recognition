import suit_preprocess
import os
import cv2
import numpy as np

# Preps image to find contours
# Gray -> blur -> thresh
def prep_gray_blur_thresh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # TODO: maybe change thresh? Dynamic?
    retval, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('a', thresh)
    #cv2.waitKey(0)
    return thresh

# Finds contours
def find_cards(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

# Chooses only highest "parent" hierarchy, which should be the edges of the cards cards, some residuals are cleaned up later
def process_contours(c, h):
    adult_contours = []
    parents = []
    # What even is this? Did I write this lol? Why is it like this???
    for i in range(len(c)):
        parents.append(h[0][i])
    for i in range(len(c)):
        if parents[i][3] == -1:  # This -1 means parent contour
            adult_contours.append(c[i])
    return adult_contours

# Gets corners from contours, filters out non-card contours using the area, the cards have a stoopid high
# area so just making sure it is above 10,000 seems to work
def get_corners(adult_contours):
    cards = []
    for i in range(len(adult_contours)):
        perimeter = cv2.arcLength(adult_contours[i], True)
        corners = cv2.approxPolyDP(adult_contours[i], 0.1 * perimeter, True)
        # Filters out "baby" contours, residuals from the image
        if cv2.contourArea(corners) > 10000:
            cards.append(corners)
    return cards

# Uses corners and affine transform to isolate out cards
def pull_out_cards(card_corners, cards):
    pulled_cards = []
    for i in card_corners:
        pts = np.squeeze(i)
        sorted_pts = sorted(pts, key=lambda x: x[0])
        #Makes sure cards are correct orientation
        if sorted_pts[0][1] > sorted_pts[1][1]:
            sorted_pts = swap_positions(sorted_pts, 0, 1)
        if sorted_pts[2][1] < sorted_pts[3][1]:
            sorted_pts = swap_positions(sorted_pts, 2, 3)
        sorted_pts = np.float32(sorted_pts)

        # TODO: Just chose 300x400, maybe measure the dimensions of the cards to get a better ratio?
        pts1_ortho = np.float32([[0, 0], [0, 400], [300, 400], [300, 0]])
        H1, _ = cv2.findHomography(srcPoints=sorted_pts, dstPoints=pts1_ortho)
        warp = cv2.warpPerspective(cards, H1, (300, 400))
        pulled_cards.append(warp)
    return pulled_cards

def swap_positions(l, pos1, pos2):
    l[pos1], l[pos2] = l[pos2], l[pos1]
    return l

def mini_thresh(warped_card):
    gray = cv2.cvtColor(warped_card, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # TODO: maybe change thresh? Dynamic?
    retval, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    # cv2.imshow('mini_thresh', thresh)
    # cv2.waitKey(0)
    return thresh

def match_rank(warped_card):
    #Set Default Card Value
    card_value = 0
    ace_flag = False

    #Setup Storage Space for analysis of all data
    result_array = []

    template = match_template(warped_card, result_array)

    #Find the highest max val and save its index. We use that index for the classification.
    #This way we identify all cards instead of outright missing one.
    chosen_index = 0
    chosen_maxVal = 0
    index = 0

    for each in result_array:
        if each[1] > chosen_maxVal:
            chosen_maxVal = each[1]
            chosen_index = index
            (f_minVal, f_maxVal, f_minLoc, f_maxLoc) = each
        index = index + 1

    if chosen_maxVal > 0.1:
        (startX, startY) = f_maxLoc
        endX = startX + template.shape[1]
        endY = startY + template.shape[0]

        cv2.rectangle(warped_card, (startX, startY), (endX, endY), (255, 0, 0), 3)

        card_value = chosen_index + 1

        if card_value == 1:
            ace_flag = True

        if card_value > 10:
            card_value = 10

        print("Found Value: " + str(card_value))

        cv2.imshow("template_out", warped_card)
        cv2.waitKey(0)
    return (card_value, ace_flag)

def match_template(warped_card, result_array):
    #Template Ace
    template = cv2.imread("rank" + "\\" + "ace.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Two
    template = cv2.imread("rank" + "\\" + "two.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Three
    template = cv2.imread("rank" + "\\" + "three.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Four
    template = cv2.imread("rank" + "\\" + "four.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Five
    template = cv2.imread("rank" + "\\" + "five.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Six
    template = cv2.imread("rank" + "\\" + "six.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Seven
    template = cv2.imread("rank" + "\\" + "seven.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Eight
    template = cv2.imread("rank" + "\\" + "eight.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Nine
    template = cv2.imread("rank" + "\\" + "nine.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Ten
    template = cv2.imread("rank" + "\\" + "ten.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Jack
    template = cv2.imread("rank" + "\\" + "jack.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template Queen
    template = cv2.imread("rank" + "\\" + "queen.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))

    #Template King
    template = cv2.imread("rank" + "\\" + "king.jpg")
    result = cv2.matchTemplate(warped_card, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    result_array.append((minVal, maxVal, minLoc, maxLoc))
    return template

def main():
    #Uncomment if suits are not processes (current: processed)
    #suit_preprocess.process()

    #Loops through images in cards file
    for root, dirs, files in os.walk("cards"):
        for pic in files:
            cards = cv2.imread("cards" + "\\" + pic)
            cards = cv2.resize(cards, (0, 0), fx=.2, fy=.2) #Resize to 20%, not set in stone
            threshed_cards = prep_gray_blur_thresh(cards)
            contours, hierarchy = find_cards(threshed_cards)
            adult_contours = process_contours(contours, hierarchy)

            #Uncomment to see contours
            #cv2.drawContours(cards, adult_contours, -1, (0, 255, 0), 3)
            #cv2.imshow('a', cards)
            #cv2.waitKey(0)

            card_corners = get_corners(adult_contours)
            pulled_cards = pull_out_cards(card_corners, cards)

            #Totals up values of cards found
            card_values = []
            total = 0
            visible_ace = False

            for card in pulled_cards:
                (value, ace_flag) = match_rank(card)

                if ace_flag:
                    visible_ace = True

                card_values.append(value)

            for each in card_values:
                total = total + each

            if visible_ace:
                if total < 11:
                    total = total + 10

            print(total)

if __name__ == "__main__":
    main()
