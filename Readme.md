# Converting Scipy Cookbook

This repo contains Scipy cookbook partially converted to sphinx format

The folders contain various stages of conversion:

* **originals** contains the wiki source and attachments scraped from
   wiki dump posted by Robert Kern
   http://mail.scipy.org/pipermail/scipy-dev/2013-May/018792.html. using
   scrape_cookbook.py
   
* **converted** Contains files converted to
   [Pweave](http://mpastell.com) rst+noweb file format converted with Pandoc.
   
* **attachments** is Pweave files with fixed attachment formatting
    that wasn't handled by Pandoc.

* **rst** Has the files converted to rst format using Pweave, notice
    that the code is not run so Pweave just formats code for Sphinx
    documents. ~30% of the examples can be run with Pweave to produce
    meaningful output with captured figures and code.

You can see the result of conversion in [here](http://mpastell.github.io)
