__author__ = 'rhrub'



import numpy as np
import h5py
from vertica_python import connect
import time

srcFolder = '/mnt/DATA/repo/AVSC/Data/'
hdf_type = '.hdf5'
target = 'testSequenceLabeled'

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
fields = 9

cur.execute('select t.id '
            'from transactions as t, trainHistory as h '
            'where t.id = h.id '
            'group by t.id having count(*) > 200 and count(*) < %s;' % limit)


ids = cur.fetchall()


ids = np.array(ids)
ids = ids.astype(np.float64)
ids = ids.reshape((-1))

print 'ids shape: ' + str(ids.shape)

#HDF5
f = h5py.File(srcFolder + target + hdf_type, 'w')
hdfData = f.create_dataset('hdfDataSet', shape=(ids.shape[0], fields, limit), maxshape=(ids.shape[0], None, None))

#hdfData[:, :] = labels[:, :]

count = 0
for i in ids:
    start = time.time()
    
    z = np.zeros((fields, limit))

    cur.execute('select t.chain, t.dept, t.category, t.company, t.brand, o.category, o.offervalue, o.brand, case when h.repeater = \'f\' then 0 else 1 end '
                'from transactions as t, trainHistory as h, offers as o '
                'where t.id = h.id and h.offer = o.offer and t.id = %s '
                'order by transdate;' % int(i))
    d = cur.fetchall()
    #print d
    d = np.array(d)
    d = d.astype(np.float32).T

    z[:d.shape[0], :d.shape[1]] = d

    d = z.reshape((1, fields, limit))

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
