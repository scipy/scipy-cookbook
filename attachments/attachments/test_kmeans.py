#!/usr/bin/env python
import matplotlib.pyplot as plt
import scipy.cluster.vq as scvq
import scipy as sp

#--------------------------------------------------------------
def test_kmeans():
	obs = sp.random.uniform(0, 10, (1000, 2))
	# knum = 7
	obs = scvq.whiten(obs)

	# run kmeans with diffirent number of clusters
	for knum in range(2, 8):
		codebook, dist = scvq.kmeans(obs, knum)
		ind, dist = scvq.vq(obs, codebook)

		# visualize
		# plt.ion()
		plt.ioff()
		plt.figure(knum)
		colors = ["b*", "g+", "ro", "yp", "ms", "ch", "wx"]

		for icluster in range(knum):
			x = (ind == icluster).nonzero()[0]
			plt.plot(obs[x, 0], obs[x, 1], colors[icluster])

			for iline in range(sp.size(x)):
				plt.plot([obs[x[iline], 0], codebook[icluster, 0]],
					[obs[x[iline], 1], codebook[icluster, 1]], "k--")

		# the cluster centroid
		plt.plot(codebook[:, 0], codebook[:, 1], "ko")

		# the plot size
		plt.xlim((-0.3, 3.8))
		plt.ylim((-0.3, 3.8))
	plt.show()
# end.def

#--------------------------------------------------------------
if __name__ == "__main__":
	test_kmeans()
