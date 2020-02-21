import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

rval, frame = vc.read()

while True:

  if frame is not None:   
     cv2.imshow("preview", frame)
  rval, frame = vc.read()
  print(rval)
  if not rval:
     raise IOError("webcam failure")
  if cv2.waitKey(1) & 0xFF == ord('q'):
     break