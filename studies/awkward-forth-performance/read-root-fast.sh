#################################################################### jagged0

python -c 'import uproot, os; print("zlib9-jagged0", uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root:tree/branch").num_entries, "entries", os.stat("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root").st_size, "bytes")'

vmtouch -t /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root > /dev/null
vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root | fgrep Pages

rm -f AutoDict_vector_vector_vector_float_____* AutoDict_vector_vector_float___* AutoDict_vector_float_* read_jagged3_root_C* read_jagged2_root_C* read_jagged0_root_C* read_jagged0_root_C*

root -l << EOF
.L read_jagged0_root.C++
read_jagged0_root()
.q
EOF

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root | fgrep Pages

python forth-read-jagged0-root.py

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged0.root | fgrep Pages

python uproot-read-jagged0-root.py

#################################################################### jagged1

python -c 'import uproot, os; print("zlib9-jagged1", uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root:tree/branch").num_entries, "entries", os.stat("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root").st_size, "bytes")'

vmtouch -t /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root > /dev/null
vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root | fgrep Pages

rm -f AutoDict_vector_vector_vector_float_____* AutoDict_vector_vector_float___* AutoDict_vector_float_* read_jagged3_root_C* read_jagged2_root_C* read_jagged1_root_C* read_jagged0_root_C*

root -l << EOF
.L read_jagged1_root.C++
read_jagged1_root()
.q
EOF

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root | fgrep Pages

python forth-read-jagged1-root.py

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged1.root | fgrep Pages

python uproot-read-jagged1-root.py

#################################################################### jagged2

python -c 'import uproot, os; print("zlib9-jagged2", uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged2.root:tree/branch").num_entries, "entries", os.stat("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged2.root").st_size, "bytes")'

vmtouch -t /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged2.root > /dev/null
vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged2.root | fgrep Pages

rm -f AutoDict_vector_vector_vector_float_____* AutoDict_vector_vector_float___* AutoDict_vector_float_* read_jagged3_root_C* read_jagged2_root_C* read_jagged1_root_C* read_jagged0_root_C*

root -l << EOF
.L read_jagged2_root.C++
read_jagged2_root()
.q
EOF

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged2.root | fgrep Pages

python forth-read-jagged2-root.py

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged2.root | fgrep Pages

python uproot-read-jagged2-root.py

#################################################################### jagged3

python -c 'import uproot, os; print("zlib9-jagged3", uproot.open("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root:tree/branch").num_entries, "entries", os.stat("/home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root").st_size, "bytes")'

vmtouch -t /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root > /dev/null
vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root | fgrep Pages

rm -f AutoDict_vector_vector_vector_float_____* AutoDict_vector_vector_float___* AutoDict_vector_float_* read_jagged3_root_C* read_jagged2_root_C* read_jagged1_root_C* read_jagged3_root_C*

root -l << EOF
.L read_jagged3_root.C++
read_jagged3_root()
.q
EOF

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root | fgrep Pages

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root | fgrep Pages

python forth-read-jagged3-root.py

vmtouch /home/jpivarski/storage/data/chep-2021-jagged-jagged-jagged/zlib9-jagged3.root | fgrep Pages

python uproot-read-jagged3-root.py
