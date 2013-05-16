``* imshow - You can get seemingly quirky behavior if you do not set vmin and vmax manually. If they are left unset, then !AxisImage will attempt to automatically scale the values of the elements in order to keep the luminance constant. However, if you are doing something like an animation, or want to compare two images against each other, this can cause problems. For example, if you know your values will range between 0 and 1, you can do: imshow(img, vmin=0, vmax=1, cmap=cm.gray, interpolation=``\ \ ``). This also sets the color map to grays, and to use square blocks for the elements.``

--------------

CategoryCookbookMatplotlib

