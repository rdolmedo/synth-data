import json, subprocess, sys, time
repo = sys.argv[1]; out = sys.argv[2]
owner, name = repo.split('/')
Q = '''query($owner:String!,$name:String!,$cursor:String,$n:Int!){repository(owner:$owner,name:$name){pullRequests(first:$n,after:$cursor,orderBy:{field:CREATED_AT,direction:ASC}){pageInfo{hasNextPage endCursor}
nodes{number title state merged createdAt closedAt mergedAt author{login} baseRefName headRefName headRepository{nameWithOwner} additions deletions changedFiles
labels(first:20){nodes{name}} reviewDecision comments{totalCount} reviews{totalCount}
files(first:100){nodes{path additions deletions changeType}}}}}}'''
cursor=None; allp=[]; n=int(sys.argv[3]) if len(sys.argv)>3 else 40
while True:
    args=['gh','api','graphql','-f',f'query={Q}','-F',f'owner={owner}','-F',f'name={name}','-F',f'n={n}']
    if cursor: args+=['-F',f'cursor={cursor}']
    for attempt in range(8):
        r=subprocess.run(args,capture_output=True,text=True)
        if r.returncode==0: break
        print('retry',r.stderr[:200],file=sys.stderr); time.sleep(5*(attempt+1))
    d=json.loads(r.stdout)['data']['repository']['pullRequests']
    allp+=d['nodes']; print(repo,len(allp),file=sys.stderr)
    if not d['pageInfo']['hasNextPage']: break
    cursor=d['pageInfo']['endCursor']
json.dump(allp,open(out,'w'))
