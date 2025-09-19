echo $CONFIGS
echo $CONFIGMAPS
echo $JOBS

for file in $(ls $CONFIGS)
do
name=$(basename -s .yaml $file)
echo $name
kubectl create configmap -n team-project $name-cm \
    --from-file=experiment.yaml=$CONFIGS/$file \
    --dry-run=client -o=yaml > $CONFIGMAPS/$name-cm.yaml
kubectl create job $name --image=busybox \
    -o=go-template-file=k8s-resources/experiments/sklearn/experiment-template.go \
    --dry-run=client > $JOBS/$name-job.yaml
done