import numpy as np

def gen_moon(rad=1.0,occ=0.5,num_samples=100):
    sampled = 0
    samples = np.zeros([num_samples,2])
    while sampled < num_samples:
        #print(samples)
        r = np.sqrt(np.random.rand())*rad
        theta = np.random.rand()*2*np.pi
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        if np.sqrt((x+occ)**2+y**2)>rad:
            samples[sampled,0] = x
            samples[sampled,1] = y
            sampled += 1
    return samples

def gen_disk(rad1=1.0,rad2=2.0,num_samples=100):
    sampled = 0
    samples = np.zeros([num_samples,2])
    while sampled < num_samples:
        #print(samples)
        r = np.sqrt(np.random.rand())*rad2
        theta = np.random.rand()*2*np.pi
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        if r>rad1:
            samples[sampled,0] = x
            samples[sampled,1] = y
            sampled += 1
    return samples

def two_moons(num_samples=600,offset=np.array([-0.5,0.8])):
    samples1 = gen_moon(num_samples=int(num_samples/2))
    samples2 = -gen_moon(num_samples=int(num_samples/2))+offset[None,:]
    return np.concatenate([samples1,samples2])

def rings(num_samples = 600, radii=[0,1,2,3]):
    num_rings = int(len(radii)/2)
    ratios = [(radii[2*k+1]**2-radii[2*k]**2)/(radii[1]**2-radii[0]**2) for k in range(num_rings)]
    basenum = num_samples/np.sum(ratios)

    samples = np.concatenate([gen_disk(rad1=radii[2*k],rad2=radii[2*k+1],num_samples=int(ratios[k]*basenum)) for k in range(num_rings)])
    return samples
