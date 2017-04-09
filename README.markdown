# CNN Viz 

Note: this is alpha software. I quickly threw this together to play with convnet visualizations on my iPhone. There are a lot of things in the code that are not quite kosher yet. It was only tested on the iPhone 6s. That said, it *should* work on other iPhones too. ;-)

## Improvements

Things that need work:

- The camera code does not handle interruptions, going to the background, etc. In a production quality app these sorts of loose ends need to be tied up.

- There may be glitches between how the frames from the video stream are sent to Metal. I did not think this through very carefully yet.

- The UI to swipe between panels needs work (some kind of visual feedback).
