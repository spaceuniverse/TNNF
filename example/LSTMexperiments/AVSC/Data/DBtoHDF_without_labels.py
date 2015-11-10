__author__ = 'rhrub'



import numpy as np
import h5py
from vertica_python import connect
import time

srcFolder = '/mnt/DATA/repo/AVSC/Data/'
hdf_type = '.hdf5'
target = 'testSequence'

#Get from DB
connection = connect({
    'host': '192.168.240.94',
    'port': 5433,
    'user': 'dbadmin',
    'password': 'vertica7',
    'database': 'Shoppers'
})

cur = connection.cursor()

limit = 4000

cur.execute('select id from transactions group by id having count(*) > 200 and count(*) < %s;' % limit)


ids = cur.fetchall()


ids = np.array(ids)
ids = ids.astype(np.float64)
ids = ids.reshape((-1))

print 'ids shape: ' + str(ids.shape)

#HDF5
f = h5py.File(srcFolder + target + hdf_type, 'w')
hdfData = f.create_dataset('hdfDataSet', shape=(ids.shape[0], 5, limit), maxshape=(ids.shape[0], None, None))

#hdfData[:, :] = labels[:, :]

count = 0
for i in ids:
    start = time.time()
    
    z = np.zeros((5, limit))

    cur.execute('select chain, dept, category, company, brand from transactions where id = %s order by transdate;' % int(i))
    d = cur.fetchall()
    d = np.array(d)
    d = d.astype(np.float32).T
    
    
    #print d.shape, 'select chain, dept, category, company, brand from transactions where id = %s order by transdate;' % int(i)

    z[:d.shape[0], :d.shape[1]] = d

    d = z.reshape((1, 5, limit))

    hdfData[count, :, :] = d[:, :]
    
    stop = time.time()

    print str(count) + '\t' + 'Its shape: ' + str(d.shape) + '\t' + 't: ' + str(round( (stop - start) , 2 )) + ' sec'
    count += 1

    if count % 10 == 0:
        print 'esimated time: ' + str(round( (stop - start) * ids.shape[0] / 3600, 2 )) + ' hrs'


connection.close()
print hdfData.shape
print hdfData[0, :, :3]
f.close()
