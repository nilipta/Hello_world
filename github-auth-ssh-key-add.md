nil-pop@pop-os:~$ cd Hello_world/
nil-pop@pop-os:~/Hello_world$ git status
On branch master
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean
nil-pop@pop-os:~/Hello_world$ git pull
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
nil-pop@pop-os:~/Hello_world$ ls -al ~/.ssh
total 24
drwx------  2 nil-pop nil-pop 4096 Mar 21 12:55 .
drwxr-x--- 32 nil-pop nil-pop 4096 May  3 22:35 ..
-rw-------  1 nil-pop nil-pop  419 Mar 21 12:52 id_ed25519
-rw-r--r--  1 nil-pop nil-pop  108 Mar 21 12:52 id_ed25519.pub
-rw-------  1 nil-pop nil-pop  978 Mar 21 12:55 known_hosts
-rw-r--r--  1 nil-pop nil-pop  142 Mar 21 12:55 known_hosts.old
nil-pop@pop-os:~/Hello_world$ eval "$(ssh-agent -s)"
Agent pid 5212
nil-pop@pop-os:~/Hello_world$ ssh-add ~/.ssh/id_ed25519
Identity added: /home/nil-pop/.ssh/id_ed25519 (ni----------hy@gmail.com)
nil-pop@pop-os:~/Hello_world$ cat ~/.ssh/id_ed25519.pub
ssh-ed25519 A-------------------------------------------------y ni-------------y@gmail.com
nil-pop@pop-os:~/Hello_world$ ssh -T git@github.com
Hi nilipta! You've successfully authenticated, but GitHub does not provide shell access.
nil-pop@pop-os:~/Hello_world$ git remote set-url origin git@github.com:ni---a/Hello_world.git
nil-pop@pop-os:~/Hello_world$ git pull
remote: Enumerating objects: 16, done.
remote: Counting objects: 100% (16/16), done.
remote: Compressing objects: 100% (12/12), done.
remote: Total 15 (delta 2), reused 15 (delta 2), pack-reused 0 (from 0)
Unpacking objects: 100% (15/15), 209.32 KiB | 245.00 KiB/s, done.
From github.com:nilipta/Hello_world
   a95121f..b477b19  master     -> origin/master
Updating a95121f..b477b19
Fast-forward
 .ipynb_checkpoints/tensorflow-checkpoint.ipynb                         |   343 +
 group policies/.V9!rT#7pLx@2qZ$8wF.txt                                 |     3 +
 group policies/gitmoji _ An emoji guide for your commit messages.mhtml |  1792 +++
 group policies/password-set-wifi.txt                                   |    41 +
 group policies/wifi-connect.mhtml                                      | 19779 ++++++++++++++++++++++++++++++++
 group policies/wifi.xml                                                |    25 +
 shared-clipboard-server.py                                             |    47 +
 shared-clipboard.html                                                  |    83 +
 twitter-infos.txt                                                      |    93 +
 9 files changed, 22206 insertions(+)
 create mode 100644 .ipynb_checkpoints/tensorflow-checkpoint.ipynb
 create mode 100644 group policies/.V-------F.txt
 create mode 100644 group policies/gitmoji _ An emoji guide for your commit messages.mhtml
 create mode 100644 group policies/password-set-wifi.txt
 create mode 100644 group policies/wifi-connect.mhtml
 create mode 100644 group policies/wifi.xml
 create mode 100644 shared-clipboard-server.py
 create mode 100644 shared-clipboard.html
 create mode 100644 twitter-infos.txt
nil-pop@pop-os:~/Hello_world$ 


- creating a new key
- ssh-keygen -t ed25519 -C "your_email@example.com"


